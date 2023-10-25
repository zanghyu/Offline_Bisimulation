import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
from transition_model import make_transition_model


class Encoder(nn.Module):
    def __init__(self, state_dim, encoder_dim):
        super(Encoder, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, encoder_dim)

    def forward(self, state):
        enc = F.relu(self.l1(state))
        enc = F.relu(self.l2(enc))
        enc = F.relu(self.l3(enc))
        return enc


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3_BC(object):
    def __init__(
            self,
            name,
            state_dim,
            action_dim,
            max_action,
            discount=0.99,
            tau=0.005,
            save_eigen=False,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            alpha=2.5,
            device=torch.device("cuda:0"),
            with_transition_model=False,
            slope=0.5,
            reward_scale=True,
    ):
        encoder_dim = 256
        self.name = name
        self.device = device
        self.encoder = Encoder(state_dim, encoder_dim).to(self.device)
        self.target_encoder = copy.deepcopy(self.encoder).to(self.device)

        self.save_eigen = save_eigen
        
        self.encoder_optimizer = torch.optim.Adam(
                list(self.encoder.parameters()), lr=3e-4)

        self.actor = Actor(encoder_dim, action_dim, max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(
            list(self.actor.parameters()), lr=3e-4)

        self.critic = Critic(encoder_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic.parameters()), lr=3e-4)

        transition_model_type = 'ensemble'
        self.transition_model = make_transition_model(
            transition_model_type, encoder_dim, action_dim, self.device
        ).to(self.device)
        decoder_weight_lambda = 0.0000001
        self.transition_opt = torch.optim.Adam(
            self.transition_model.parameters(), lr=3e-4, weight_decay=decoder_weight_lambda
        )

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.alpha = alpha
        
        self.with_transition_model = with_transition_model
        self.slope = float(slope)
        self.reward_scale = reward_scale
        
        self.total_it = 0
        self.it = 0

        self.effective_eigen = []

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        state = self.encoder(state)
        return self.actor(state).cpu().data.numpy().flatten()

    def soft_update_params(self, net, target_net, tau):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(
                tau * param.data + (1 - tau) * target_param.data
            )

    def update_transition_model(self, state, action, next_state):
        h = self.encoder(state)
        pred_next_latent_mu, pred_next_latent_sigma = self.transition_model(
            torch.cat([h, action], dim=1))
        if pred_next_latent_sigma is None:
            pred_next_latent_sigma = torch.ones_like(pred_next_latent_mu)
        pred_next_latent_sigma = pred_next_latent_sigma.float()
        next_h = self.encoder(next_state)
        diff = (pred_next_latent_mu - next_h.detach()) / pred_next_latent_sigma
        diff = diff.float()
        loss = torch.mean(0.5 * diff.pow(2) +
                          torch.log(pred_next_latent_sigma))
        return loss

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1
        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        if self.total_it < 1e5:
            z_x = self.encoder(state)    # (batch_size, feature_dim)
            batch_size = state.size(0)
            if self.with_transition_model:
                with torch.no_grad():
                    z_x_tar = self.target_encoder(state)
                    pred_z_x_next = self.transition_model.sample_prediction(
                        torch.cat([z_x_tar, action], dim=1))
            else:
                with torch.no_grad():
                    z_x_tar = self.target_encoder(state)
                    pred_z_x_next = self.target_encoder(next_state)
            
            def cosine_dis(x, y):
                numerator = torch.sum(x * y)
                denominator = torch.sqrt(torch.sum(x**2)) * torch.sqrt(torch.sum(y**2))
                cos_similarity = numerator / (denominator + 1e-9)
                return torch.atan2(torch.sqrt(torch.maximum(1.-cos_similarity**2, torch.tensor(0.))), cos_similarity)


            def squarify(x):
                batch_size = x.shape[0]
                if len(x.shape) > 1:
                    representation_dim = x.shape[-1]
                    return torch.reshape(torch.tile(x, [batch_size]),
                                    (batch_size, batch_size, representation_dim))
                return torch.reshape(torch.tile(x, [batch_size]), (batch_size, batch_size))

            def diff(z1, z2, beta=0.1):
                batch_size = z1.shape[0]
                rep_dim = z1.shape[-1]

                z1 = squarify(z1)
                z2 = squarify(z2)
                first_squared_reps = torch.reshape(z1, [batch_size**2, rep_dim])
                second_squared_reps = z2.permute([1,0,2])
                second_squared_reps = torch.reshape(second_squared_reps, [batch_size**2, rep_dim])
                z_cosine_dis_ = cosine_dis(first_squared_reps, second_squared_reps)
                norm_average = 0.5*(torch.sum(torch.square(first_squared_reps),-1)+torch.sum(torch.square(second_squared_reps),-1))

                z_diffs = norm_average + beta * z_cosine_dis_
                return z_diffs

            
            z_diff = diff(z_x, z_x_tar)
            squared_rews = squarify(reward)
            squared_rews_transp = squared_rews.permute([1,0,2])
            squared_rews = squared_rews.reshape(squared_rews.shape[0]**2,1)
            squared_rews_transp = squared_rews_transp.reshape(squared_rews_transp.shape[0]**2,1)
            r_diff = torch.abs(squared_rews - squared_rews_transp)

            if self.reward_scale:
                r_diff *= (1 - self.discount)

            with torch.no_grad():
                next_diff = diff(pred_z_x_next, pred_z_x_next)

            bisimilarity = r_diff + self.discount * next_diff

            def expectile_loss(diff):
                weight = torch.where(diff > 0, self.slope, (1 - self.slope))
                return weight * (diff ** 2)
            encoder_loss = expectile_loss(bisimilarity.detach() - z_diff)
            enc_loss = encoder_loss.mean()
            
            if self.with_transition_model:
                transition_loss = self.update_transition_model(state, action, next_state)
                self.encoder_optimizer.zero_grad(set_to_none=True)
                self.transition_opt.zero_grad()
                (enc_loss + transition_loss).backward()
                torch.nn.utils.clip_grad_norm(self.transition_model.parameters(), 0.25)
                self.encoder_optimizer.step()
                self.transition_opt.step()
            else:
                self.encoder_optimizer.zero_grad(set_to_none=True)
                enc_loss.backward()
                self.encoder_optimizer.step()
            
            self.soft_update_params(self.encoder, self.target_encoder, tau=0.05)
            return

        state_ = self.encoder(state).detach()
        next_state_ = self.encoder(next_state).detach()

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                    torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            na = self.actor_target(next_state_)
            next_action = (
                    na + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state_, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state_, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + \
                      F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor loss
            state_ = self.encoder(state).detach()
            pi = self.actor(state_)
            Q = self.critic.Q1(state_, pi)
            lmbda = self.alpha / Q.abs().mean().detach()

            actor_loss = - lmbda * Q.mean() + F.mse_loss(pi, action)

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)

        if (self.total_it % self.policy_freq == 0) and (self.total_it % 10000 == 0):
            print(actor_loss)
            print(critic_loss)

    def save(self, filename):
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(),
                   filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(),
                   filename + "_actor_optimizer")

        torch.save(self.encoder.state_dict(), filename + "_encoder")
        #torch.save(self.predictor.state_dict(), filename + "_predictor")
        torch.save(self.encoder_optimizer.state_dict(),
                   filename + "_encoder_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(
            torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(
            torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)

        self.encoder.load_state_dict(torch.load(filename + "_encoder"))
        self.encoder_optimizer.load_state_dict(
            torch.load(filename + "_encoder_optimizer"))

        #self.predictor.load_state_dict(torch.load(filename + "_predictor"))

    def load_enc(self, filename):
        self.encoder.load_state_dict(torch.load(filename + "_encoder"))
        #self.predictor.load_state_dict(torch.load(filename + "_predictor"))
