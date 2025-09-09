"""
PATE-GAN implementation by BorealisAI.
source: https://github.com/BorealisAI/private-data-generation/blob/737df84e3f1ee521190cc2b62ce408ad708206e6/models/pate_gan.py
"""

# Copyright 2019 RBC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# pate_gan.py implements the PATE_GAN generative model to generate private synthetic data


import math
import numpy as np
# from scipy.special import expit, logit
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils


class Generator(nn.Module):
    def __init__(self, latent_size, output_size, conditional=True):
        super().__init__()
        z = latent_size
        d = output_size
        if conditional:
            z = z + 1
        else:
            d = d + 1
        self.main = nn.Sequential(
            nn.Linear(z, 2 * latent_size),
            nn.ReLU(),
            nn.Linear(2 * latent_size, d))

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, input_size, wasserstein=False):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(input_size + 1, int(input_size / 2)),
            nn.ReLU(),
            nn.Linear(int(input_size / 2), 1))

        if not wasserstein:
            self.main.add_module(str(3), nn.Sigmoid())

    def forward(self, x):
        return self.main(x)


def weights_init(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def pate(data, netTD, lap_scale):
    results = torch.Tensor(len(netTD), data.size()[0]).type(torch.int64)
    for i in range(len(netTD)):
        output = netTD[i].forward(data)
        pred = (output > 0.5).type(torch.Tensor).squeeze()
        results[i] = pred

    clean_votes = torch.sum(results, dim=0).unsqueeze(1).type(torch.DoubleTensor)
    noise = torch.from_numpy(np.random.laplace(loc=0, scale=1 / lap_scale, size=clean_votes.size()))  # .cuda()
    noisy_results = clean_votes + noise
    noisy_labels = (noisy_results > len(netTD) / 2).type(torch.DoubleTensor)

    return noisy_labels, clean_votes


def moments_acc(num_teachers, clean_votes, lap_scale, l_list):
    q = (2 + lap_scale * torch.abs(2 * clean_votes - num_teachers)
         ) / (4 * torch.exp(lap_scale * torch.abs(2 * clean_votes - num_teachers)))

    update = []
    for l in l_list:
        a = 2 * lap_scale * lap_scale * l * (l + 1)
        t_one = (1 - q) * torch.pow((1 - q) / (1 - math.exp(2 * lap_scale) * q), l)
        t_two = q * torch.exp(2 * lap_scale * l)
        # BUG!!!
        t = t_one + t_two
        update.append(torch.clamp(t, max=a).sum())

    return torch.DoubleTensor(update)


class PG_BORAI:
    def __init__(self, X_shape, z_dim=None, num_teachers=10, epsilon=8, delta=1e-5, max_iter=10000, conditional=False):
        self.input_dim = X_shape[1] - 1
        if z_dim is None:
            self.z_dim = int(self.input_dim / 4 + 1) if self.input_dim % 4 == 0 else int(self.input_dim / 4)
        else:
            self.z_dim = z_dim
        self.generator = Generator(self.z_dim, self.input_dim, conditional).double()  # .cuda().double()
        self.student_disc = Discriminator(self.input_dim, wasserstein=False).double()  # .cuda().double()
        self.teacher_disc = [Discriminator(self.input_dim, wasserstein=False).double()  # .cuda().double()
                             for _ in range(num_teachers)]
        self.generator.apply(weights_init)
        self.student_disc.apply(weights_init)
        self.num_teachers = num_teachers
        for i in range(num_teachers):
            self.teacher_disc[i].apply(weights_init)

        self.epsilon = epsilon
        self.delta = delta
        self.max_iter = max_iter
        self.conditional = conditional

    def fit(self, x_train, lr=1e-4, batch_size=64, num_teacher_iters=5, num_student_iters=5, num_moments=100, lap_scale=0.0001, class_ratios=None):
        # Prerocess data
        # source: https://github.com/BorealisAI/private-data-generation/blob/737df84e3f1ee521190cc2b62ce408ad708206e6/evaluate.py#L126
        # x_train = expit(x_train) -- this gives lots of NaNs
        self.processor = MinMaxScaler(clip=True)
        x_train = self.processor.fit_transform(x_train)

        x_train, y_train = x_train[:, :-1], x_train[:, -1]

        batch_size = min(batch_size, len(x_train) // self.num_teachers)

        class_ratios = None
        if self.conditional:
            class_ratios = torch.from_numpy(class_ratios)

        real_label = 1
        fake_label = 0

        alpha = torch.DoubleTensor([0.0 for _ in range(num_moments)])
        l_list = 1 + torch.DoubleTensor(range(num_moments))

        criterion = nn.BCELoss()
        optimizer_g = optim.Adam(self.generator.parameters(), lr=lr)
        optimizer_sd = optim.Adam(self.student_disc.parameters(), lr=lr)
        optimizer_td = [optim.Adam(self.teacher_disc[i].parameters(), lr=lr) for i in range(self.num_teachers)]

        tensor_data = data_utils.TensorDataset(torch.DoubleTensor(x_train), torch.DoubleTensor(y_train))

        train_loader = []
        for teacher_id in range(self.num_teachers):
            start_id = teacher_id * len(tensor_data) / self.num_teachers
            end_id = (teacher_id + 1) * len(tensor_data) / self.num_teachers if teacher_id != (self.num_teachers - 1) else len(tensor_data)
            train_loader.append(data_utils.DataLoader(torch.utils.data.Subset(tensor_data, range(int(start_id), int(end_id))), batch_size=batch_size, shuffle=True))

        steps = 0
        self.epsilon_hat = 0

        while self.epsilon_hat < self.epsilon and steps < self.max_iter:

            # train the teacher discriminators
            for t_2 in range(num_teacher_iters):
                for i in range(self.num_teachers):
                    inputs, categories = None, None
                    for b, data in enumerate(train_loader[i], 0):
                        inputs, categories = data
                        break

                    # train with real
                    optimizer_td[i].zero_grad()
                    label = torch.full((inputs.size()[0],), real_label)  # .cuda()
                    output = self.teacher_disc[i].forward(torch.cat([inputs, categories.unsqueeze(1).double()], dim=1))
                    label = label.unsqueeze(1)
                    label = label.double()
                    err_d_real = criterion(output, label)
                    err_d_real.backward()

                    # train with fake
                    z = torch.Tensor(batch_size, self.z_dim).uniform_(0, 1)  # .cuda()
                    label.fill_(fake_label)

                    if self.conditional:
                        category = torch.multinomial(class_ratios, inputs.size()[0], replacement=True).unsqueeze(1).double()  # .cuda().double()
                        fake = self.generator(torch.cat([z.double(), category], dim=1))
                        output = self.teacher_disc[i].forward(torch.cat([fake.detach(), category], dim=1))
                    else:
                        fake = self.generator(z.double())
                        output = self.teacher_disc[i].forward(fake)

                    err_d_fake = criterion(output, label.double())
                    err_d_fake.backward()
                    optimizer_td[i].step()

            # train the student discriminator
            for t_3 in range(num_student_iters):
                z = torch.Tensor(batch_size, self.z_dim).uniform_(0, 1)  # .cuda()

                if self.conditional:
                    category = torch.multinomial(class_ratios, inputs.size()[0], replacement=True).unsqueeze(1).double()  # .cuda().double()
                    fake = self.generator(torch.cat([z.double(), category], dim=1))
                    predictions, clean_votes = pate(torch.cat([fake.detach(), category], dim=1), self.teacher_disc, lap_scale)
                    outputs = self.student_disc.forward(torch.cat([fake.detach(), category], dim=1))
                else:
                    fake = self.generator(z.double())
                    predictions, clean_votes = pate(fake.detach(), self.teacher_disc, lap_scale)
                    outputs = self.student_disc.forward(fake.detach())

                # update the moments
                alpha = alpha + moments_acc(self.num_teachers, clean_votes, lap_scale, l_list)

                # update student
                err_sd = criterion(outputs, predictions)

                optimizer_sd.zero_grad()
                err_sd.backward()
                optimizer_sd.step()

            # train the generator
            optimizer_g.zero_grad()
            z = torch.Tensor(batch_size, self.z_dim).uniform_(0, 1)  # .cuda()
            label = torch.full((inputs.size()[0],), real_label)  # .cuda()

            if self.conditional:
                category = torch.multinomial(class_ratios, inputs.size()[0], replacement=True).unsqueeze(1).double()  # .cuda().double()
                fake = self.generator(torch.cat([z.double(), category], dim=1))
                output = self.student_disc(torch.cat([fake, category.double()], dim=1))
            else:
                fake = self.generator(z.double())
                output = self.student_disc.forward(fake)
            label = label.unsqueeze(1)
            label = label.double()
            err_g = criterion(output, label)
            err_g.backward()
            optimizer_g.step()

            # Calculate the current privacy cost
            self.epsilon_hat = min((alpha - math.log(self.delta)) / l_list)
            # if steps % 100 == 0:
            #     print("Step : ", steps, "Loss SD : ", err_sd.item(), "Loss G : ", err_g.item(), "Epsilon : ", self.epsilon_hat.item())

            steps += 1

    def generate(self, num_rows, class_ratios=None, batch_size=1000):
        steps = num_rows // batch_size
        synthetic_data = []
        if self.conditional:
            class_ratios = torch.from_numpy(class_ratios)
        for step in range(steps):
            noise = torch.randn(batch_size, self.z_dim)  # .cuda()
            if self.conditional:
                cat = torch.multinomial(class_ratios, batch_size, replacement=True).unsqueeze(1).double()  # .cuda().double()
                synthetic = self.generator(torch.cat([noise.double(), cat], dim=1))
                synthetic = torch.cat([synthetic, cat], dim=1)

            else:
                synthetic = self.generator(noise.double())

            synthetic_data.append(synthetic.cpu().data.numpy())

        if steps * batch_size < num_rows:
            noise = torch.randn(num_rows - steps * batch_size, self.z_dim)  # .cuda()

            if self.conditional:
                cat = torch.multinomial(class_ratios, num_rows - steps * batch_size, replacement=True).unsqueeze(1).double()  # .cuda().double()
                synthetic = self.generator(torch.cat([noise.double(), cat], dim=1))
                synthetic = torch.cat([synthetic, cat], dim=1)
            else:
                synthetic = self.generator(noise.double())
            synthetic_data.append(synthetic.cpu().data.numpy())

        synthetic_data = np.concatenate(synthetic_data)

        # Renormalization
        # synthetic_data = logit(synthetic_data) -- this gives lots of NaNs
        # manual fix -- o/w returns NaNs
        synthetic_data = np.clip(synthetic_data, 0, 1)
        synthetic_data = self.processor.inverse_transform(synthetic_data)
        return synthetic_data
