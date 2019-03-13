#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------
# Author:   sebastian.piec@
# Modified: 2019
# ----------------------------------------------------------------------

"""Implementation of the Pong with PyGame.
"""

import collections
import time
import weakref

import numpy as np

import math
import random

import pygame
from pygame.locals import *

MAX_POINTS = 20
BOARD_W = 300
BOARD_H = 200

# ----------------------------------------------------------------------
class Player(pygame.sprite.Sprite):
    STEP = 5
    MARGIN = STEP

    def __init__(self):
        super(Player, self).__init__()
        
        self.image = pygame.image.load("images/bar.png").convert()
        self.image.set_colorkey((255, 255, 255), RLEACCEL)

        self.rect = self.image.get_rect(center=(BOARD_W * 0.5,
                                                BOARD_H * 0.9))

    def keyboard_action(self, pressed_keys):
        dx = 0
        if pressed_keys[K_LEFT]:
            dx = -self.STEP
        if pressed_keys[K_RIGHT]:
            dx = self.STEP
        self.update_position(dx)

    def external_action(self, action_idx):
        dx = 0
        if action_idx == 0:
            dx = -self.STEP
        elif action_idx == 1:
            dx = self.STEP
        self.update_position(dx)

    def update_position(self, dx):
        x = self.rect.x
        if (x + dx > self.MARGIN and
            x + self.rect.width + dx < (BOARD_W - self.MARGIN)):
            self.rect.move_ip(dx, 0)

# ----------------------------------------------------------------------
class Ball(pygame.sprite.Sprite):
    MIN_V = 5
    MAX_V = 5
    MARGIN = 5
    LOSS_REWARD = -1
    RUNNING_REWARD = 0

    def __init__(self, player):
        super(Ball, self).__init__()

        self.player = weakref.ref(player)

        self.image = pygame.image.load("images/ball.png").convert()
        self.image.set_colorkey((255, 255, 255), RLEACCEL)
        
        # random start point and velocity
        x = np.random.randint(int(BOARD_W * 0.4), int(BOARD_W * 0.6))
        y = BOARD_H * 0.2
        self.rect = self.image.get_rect(center=(x, y))

        self.vx = -self.MAX_V
        self.vx = (-1 if np.random.rand() < 0.5 else 1) * self.MAX_V
        self.vy = -self.MAX_V

    def update(self):
        """
        Returns:
            reward (float)
            is_done (bool)
        """
        x, y = self.rect.x, self.rect.y 
        if y < self.MARGIN:
            self.vy *= -1

        if (x < self.MARGIN) or (x + self.rect.width > BOARD_W - self.MARGIN):
            self.vx *= -1

        if self.rect.colliderect(self.player().rect):
            self.vy = -self.MAX_V
            self.rect.move_ip(self.vx, self.vy)
            return 1. / MAX_POINTS, False

        if self.rect.bottom > BOARD_H:
            self.ball_lost()
            return self.LOSS_REWARD, True
        self.rect.move_ip(self.vx, self.vy)

        return self.RUNNING_REWARD, False

    def ball_lost(self):
        print("Ball lost!")
        self.kill()

# ----------------------------------------------------------------------
class PongGame(object):
    #MAX_BALLS = 2               # K non-interacting balls possible
    FPS = 40

    def __init__(self, render=False):
        super(PongGame, self).__init__()

        self.render = render
        pygame.init()

        self.screen = pygame.display.set_mode((BOARD_W, BOARD_H))
        self.clock = pygame.time.Clock()

        self.player = Player()
        self.ball = None        #Ball((self.BOARD_W, self.BOARD_H), self.player)

        self.background = pygame.Surface(self.screen.get_size())
        self.background.fill((130, 210, 250))

        self.all_sprites = pygame.sprite.Group()
        self.all_sprites.add(self.player)

        pygame.font.init()
        self.header_font = pygame.font.SysFont("Arial", 20)

    def tick(self, action_idx=None):
        """If action index is None keyboard steering expected.
        """
        reward = 0
        is_done = False

        quit_requested = self._process_events()
        if quit_requested:
            return reward, True, True

        if self.render:
            self.screen.blit(self.background, (0, 0))

        if action_idx is None:
            pressed_keys = pygame.key.get_pressed()
            self.player.keyboard_action(pressed_keys)            # action_idx
        else:
            self.player.external_action(action_idx)

        if hasattr(self, "ball"):
            reward, is_done = self.ball.update()

        if self.render:
            for entity in self.all_sprites:
                self.screen.blit(entity.image, entity.rect)
            pygame.display.flip()
            self.clock.tick(self.FPS)

        return reward, is_done, False

    def _process_events(self):
        for event in pygame.event.get():
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    return True
                if event.key == K_SPACE:
                    self._new_ball()

            elif event.type == QUIT:
                return True
        return False

    def player_action(self, action_idx):
        """Externally steered bar.
        """
        reward, is_done, quit_requested = self.tick(action_idx)
        new_state = self.get_state()

        if reward == -1 or reward == 1:       
            self.all_sprites.remove(self.ball)
            self.ball.kill()
        
                # refresh screen?
            for entity in self.all_sprites:
                self.screen.blit(entity.image, entity.rect)

            del self.ball

        return reward, new_state, is_done

    def run(self):
        self.reset()

        running = True
        while running:
            _, is_done, quit_requested = self.tick()
            running = not quit_requested

            self.clock.tick(25)         # limits FPS

    def reset(self):
        self.ball_hits = 0
        self._new_ball()
        return self._numeric_state()

    def get_state(self):
        return self._numeric_state()

    def state_size(self):
        return 6

    def n_actions(self):
        return 3

    def _new_ball(self):
        self.ball = Ball(self.player)
        self.all_sprites.add(self.ball)

    def _numeric_state(self):
        state = np.zeros(self.state_size())

        state[0] = float(self.player.rect.x) / BOARD_W
        state[1] = float(self.player.rect.y) / BOARD_H

        state[2] = float(self.ball.rect.x) / BOARD_W - state[0]             # relative pos
        state[3] = float(self.ball.rect.y) / BOARD_H - state[1]
        state[4] = float(self.ball.vx) / self.ball.MAX_V
        state[5] = float(self.ball.vy) / self.ball.MAX_V
        print(state)

        return state

    def _visual_state(self):
        # e.g stack of 4 subsequent frames TODO
        pass

# ----------------------------------------------------------------------
if __name__ == "__main__":
    game = PongGame()
    game.run()