from torch.utils.data.dataset import Dataset
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO, A2C
import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import gym


class ExpertDataSet(Dataset):
    def __init__(self, expert_observations, expert_actions):
        self.observations = expert_observations
        self.actions = expert_actions

    def __getitem__(self, index):
        return self.observations[index], self.actions[index]

    def __len__(self):
        return len(self.observations)


def pretrain_agent(
    student,
    env,
    train_expert_dataset,
    test_expert_dataset,
    batch_size=64,
    epochs=1000,
    scheduler_gamma=0.7,
    learning_rate=1.0,
    log_interval=100,
    no_cuda=True,
    seed=1,
    test_batch_size=64,
):
    use_cuda = not no_cuda and th.cuda.is_available()
    th.manual_seed(seed)
    device = th.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    if isinstance(env.action_space, gym.spaces.Box):
      criterion = nn.MSELoss()
    else:
      criterion = nn.CrossEntropyLoss()

    # Extract initial policy
    model = student.policy.to(device)

    def train(model, device, train_loader, optimizer):
        model.train()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            if isinstance(env.action_space, gym.spaces.Box):
              # A2C/PPO policy outputs actions, values, log_prob
              # SAC/TD3 policy outputs actions only
              if isinstance(student, (A2C, PPO)):
                action, _, _ = model(data)
              else:
                # SAC/TD3:
                action = model(data)
              action_prediction = action.double()
            else:
              # Retrieve the logits for A2C/PPO when using discrete actions
              latent_pi, _, _ = model._get_latent(data)
              logits = model.action_net(latent_pi)
              action_prediction = logits
              target = target.long()

            loss = criterion(action_prediction, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epochs,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )
    def test(model, device, test_loader):
        model.eval()
        test_loss = 0
        with th.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)

                if isinstance(env.action_space, gym.spaces.Box):
                    # A2C/PPO policy outputs actions, values, log_prob
                    # SAC/TD3 policy outputs actions only
                    if isinstance(student, (A2C, PPO)):
                        action, _, _ = model(data)
                    else:
                        # SAC/TD3:
                        action = model(data)
                        action_prediction = action.double()
                else:
                    # Retrieve the logits for A2C/PPO when using discrete actions
                    latent_pi, _, _ = model._get_latent(data)
                    logits = model.action_net(latent_pi)
                    action_prediction = logits
                    target = target.long()

                test_loss = criterion(action_prediction, target)
        test_loss /= len(test_loader.dataset)
        print(f"Test set: Average loss: {test_loss:.4f}")

        # Here, we use PyTorch `DataLoader` to our load previously created `ExpertDataset` for training
        # and testing
        train_loader = th.utils.data.DataLoader(
            dataset=train_expert_dataset, batch_size=batch_size, shuffle=True, **kwargs
        )
        test_loader = th.utils.data.DataLoader(
            dataset=test_expert_dataset, batch_size=test_batch_size, shuffle=True, **kwargs,
        )

        # Define an Optimizer and a learning rate schedule.
        optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)
        scheduler = StepLR(optimizer, step_size=1, gamma=scheduler_gamma)

        # Now we are finally ready to train the policy model.
        for epoch in range(1, epochs + 1):
            train(model, device, train_loader, optimizer)
            test(model, device, test_loader)
            scheduler.step()

        # Implant the trained policy network back into the RL student agent
        student.policy = model
