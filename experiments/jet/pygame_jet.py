#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------
# Author:   sebastian.piec@
# Modified: 2019, March 11
# ----------------------------------------------------------------------

"""
Credits:
    https://realpython.com/pygame-a-primer/
"""

import os

import collections
import time
import numpy as np

import math
import random

import pygame
from pygame.locals import *

BOARD_W = 400
BOARD_H = 300

HIT_REWARD = -1
MISSILE_LEFT_REWARD = 0.005

# ----------------------------------------------------------------------
class Jet(pygame.sprite.Sprite):
    STEP = 5 
    MARGIN = STEP

    def __init__(self):
        super(Jet, self).__init__()
        
        self.image = pygame.image.load("images/jet.png").convert()
        self.image.set_colorkey((255, 255, 255), RLEACCEL)
        self.rect = self.image.get_rect(center=(BOARD_W * 0.15,
                                                BOARD_H * 0.5))

    def keyboard_action(self, pressed_keys):
        dx, dy = 0, 0
        if pressed_keys[K_UP]:
            dy = -self.STEP
        if pressed_keys[K_DOWN]:
            dy = self.STEP
        if pressed_keys[K_LEFT]:
            dx = -self.STEP
        if pressed_keys[K_RIGHT]:
            dx = self.STEP
        self.update_position(dx, dy)

    def external_action(self, action_idx):
        # map action's index to dx, dy
        dx, dy = 0, 0
        if action_idx == 0:
            dx = -self.STEP
        elif action_idx == 1:
            dx = self.STEP
        elif action_idx == 2:
            dy = -self.STEP
        elif action_idx == 3:
            dy = self.STEP
        self.update_position(dx, dy)

    def update_position(self, dx, dy):
        x = self.rect.x
        if (x + dx > self.MARGIN and
            x + self.rect.width + dx < (BOARD_W - self.MARGIN)):
            self.rect.move_ip(dx, 0)
        
        y = self.rect.y
        if (y + dy > self.MARGIN and
            y + self.rect.height + dy < (BOARD_H - self.MARGIN)):
            self.rect.move_ip(0, dy)

# ----------------------------------------------------------------------
class Missile(pygame.sprite.Sprite):
    MIN_SPEED = 5
    MAX_SPEED = 5

    def __init__(self):
        super(Missile, self).__init__()

        self.image = pygame.image.load("images/missile.png").convert()
        self.image.set_colorkey((255, 255, 255), RLEACCEL)
        
        x = BOARD_W - 5
        y = np.random.randint(BOARD_H)

        self.rect = self.image.get_rect(center=(x, y))
        self.vx = -random.randint(self.MIN_SPEED, self.MAX_SPEED)
        self.vy = 0
        
    def update(self):
        self.rect.move_ip(self.vx, self.vy)

        if self.rect.right <= 0:        # or self.rect.bottom > self.board_shape[1]:
            self.kill()
            return MISSILE_LEFT_REWARD
        return 0

# ----------------------------------------------------------------------
class Cloud(pygame.sprite.Sprite):
    def __init__(self):
        super(Cloud, self).__init__()
        self.image = pygame.image.load('images/cloud.png').convert()
        self.image.set_colorkey((0, 0, 0), RLEACCEL)
        self.rect = self.image.get_rect(
            center=(random.randint(BOARD_W - 40, BOARD_W), random.randint(0, BOARD_H))
        )

    def update(self):
        self.rect.move_ip(-2, 0)
        if self.rect.right < 0:
            self.kill()

# ----------------------------------------------------------------------
class Explosion(pygame.sprite.Sprite):

    def __init__(self, center, size, explosion_anim):
        pygame.sprite.Sprite.__init__(self)

        self.explosion_anim = explosion_anim

        self.size = size
        self.image = self.explosion_anim[self.size][0]
        self.rect = self.image.get_rect()
        self.rect.center = center
        self.frame = 0
        self.last_update = pygame.time.get_ticks()
        self.frame_rate = 25

    def update(self):
        now = pygame.time.get_ticks()
        if now - self.last_update > self.frame_rate:
            self.last_update = now
            self.frame += 1
            if self.frame == len(self.explosion_anim[self.size]):
                self.kill()
            else:
                center = self.rect.center
                self.image = self.explosion_anim[self.size][self.frame]
                self.rect = self.image.get_rect()
                self.rect.center = center

# ----------------------------------------------------------------------
class JetGame(object):
    MAX_MISSILES = 3

    ADDMISSILE = pygame.USEREVENT + 1
    ADDCLOUD = pygame.USEREVENT + 2
    
    def __init__(self, render=False):
        super(JetGame, self).__init__()

        self.render = render
        pygame.init()

        self.screen = pygame.display.set_mode((BOARD_W, BOARD_H))
        self.clock = pygame.time.Clock()

        self.background = pygame.Surface(self.screen.get_size())
        self.background.fill((130, 210, 250))

        self.player = Jet()

        self.all_sprites = pygame.sprite.Group()
        self.missiles = pygame.sprite.Group()
        self.explosions = pygame.sprite.Group()
        self.clouds = pygame.sprite.Group()

        self._init_explosion()

        pygame.time.set_timer(self.ADDMISSILE, 50)
        pygame.time.set_timer(self.ADDCLOUD, 3000)
        
        pygame.font.init()
        self.myfont = pygame.font.SysFont('Comic Sans MS', 30)

        self.n_missiles = 0
        self.n_hits = 0

    def tick(self, action_idx=None):
        """If action index is None keyboard steering expected.
        """
        reward = 0
        is_done = False

        quit_requested = self._process_events()             # e.g. close window event
        if quit_requested:
            return reward, True, True

        if self.render:
            self.screen.blit(self.background, (0, 0))

        if hasattr(self, "player"):
            if action_idx is not None:
                self.player.external_action(action_idx)
            else:
                pressed_keys = pygame.key.get_pressed()
                self.player.keyboard_action(pressed_keys)            # action_idx

        # update all sprites
        for missile in self.missiles:
            m_reward = missile.update()
            if m_reward == MISSILE_LEFT_REWARD:
                self.n_missiles += 1

                self.all_sprites.remove(missile)
                self.missiles.remove(missile)

            reward += m_reward

        for expl in self.explosions:
            expl.update()

        for cloud in self.clouds:
            cloud.update()

        if self.render:
            for entity in self.all_sprites:
                self.screen.blit(entity.image, entity.rect)

        if hasattr(self, "player") and pygame.sprite.spritecollideany(self.player, self.missiles):
            reward += HIT_REWARD
            is_done = True
            
            if self.render:
                expl = Explosion(self.player.rect.center, 'lg', self.explosion_anim)
                self.explosions.add(expl)
                self.all_sprites.add(expl)

            self.all_sprites.remove(self.player)
            self.player.kill()
            del self.player

            self.n_hits += 1

        if self.render:
            if self.n_missiles > 0:
                textsurface = self.myfont.render('missiles {}, hits {}, hit rate {:.2f}'.format(self.n_missiles,
                        self.n_hits, float(self.n_hits) / self.n_missiles), False, (0, 0, 0))
                self.screen.blit(textsurface,(0,0))

            pygame.display.flip()

        return reward, is_done, False

    def _process_events(self):
        for event in pygame.event.get():
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    return True
                if event.key == K_SPACE:
                    self.reset()
            elif event.type == self.ADDMISSILE:
                if len(self.missiles) < self.MAX_MISSILES:
                    missile = Missile()
                    self.missiles.add(missile)
                    self.all_sprites.add(missile)
            elif event.type == self.ADDCLOUD:
                new_cloud = Cloud()
                self.all_sprites.add(new_cloud)
                self.clouds.add(new_cloud)
            elif event.type == QUIT:
                return True
        return False

    def player_action(self, action_idx):
        # jet is steered externally, e.g. by a deep NN
        reward, is_done, quit_requested = self.tick(action_idx)
        new_state = self.get_state()

        if self.render:
            self.clock.tick(60)

        return reward, new_state, is_done

    def run(self):
        self.reset()

        while True:
            reward, is_done, quit_requested = self.tick()
            print("Reward {}, done {}, quit {}".format(reward, is_done, quit_requested))
            self.clock.tick(25)         # limits FPS
            if quit_requested:
                break

    def reset(self):
        self.player = Jet()
        self.all_sprites.add(self.player)
        return self._numeric_state()

    def get_state(self):
        return self._numeric_state()

    def state_size(self):
        return 2 + self.MAX_MISSILES * 4

    def n_actions(self):
        return 5            # left/right/up/down/do nothing

    def _numeric_state(self):
        # state being returned should posses Markov property
        state = np.zeros(self.state_size())

        # normalize state
        if hasattr(self, "player"):
            jet_x = float(self.player.rect.x) / BOARD_W
            jet_y = float(self.player.rect.y) / BOARD_H
        else:
            jet_x, jet_y = 0, 0

        state[0] = jet_x
        state[1] = jet_y
        offset = 2

        for idx, m in enumerate(self.missiles):
            state[4 * idx + offset] = float(m.rect.x) / BOARD_W - jet_x
            state[4 * idx + offset + 1] = float(m.rect.y) / BOARD_H - jet_y
            state[4 * idx + offset + 2] = float(m.vx) / m.MAX_SPEED
            state[4 * idx + offset + 3] = float(m.vy) / m.MAX_SPEED
        return state

    def _visual_state(self):
        # stack of K subsequent frames TODO
        pass

    def _init_explosion(self):
        img_dir = "images"
        explosion_anim = {}
        explosion_anim['lg'] = []
        explosion_anim['sm'] = []
        for i in range(9):
            filename = 'regularExplosion0{}.png'.format(i)
            img = pygame.image.load(os.path.join(img_dir, filename)).convert()
            img.set_colorkey((0, 0, 0), RLEACCEL)
            img_lg = pygame.transform.scale(img, (75, 75))
            explosion_anim['lg'].append(img_lg)
            img_sm = pygame.transform.scale(img, (32, 32))
            explosion_anim['sm'].append(img_sm)
        self.explosion_anim = explosion_anim

# ----------------------------------------------------------------------
if __name__ == "__main__":
    game = JetGame(render=True)
    game.run()