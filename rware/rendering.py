"""
2D rendering of the Robotic's Warehouse
environment using pyglet
"""

import math
import os
import sys

import numpy as np

from rware.warehouse import Warehouse, Direction, Point

if "Apple" in sys.version:
    if "DYLD_FALLBACK_LIBRARY_PATH" in os.environ:
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] += ":/usr/lib"
        # (JDS 2016/04/15): avoid bug on Anaconda 2.3.0 / Yosemite

import pyglet
from pyglet.graphics import Batch
from pyglet.window import Window
from pyglet.shapes import Rectangle, Line, Triangle
from pyglet.gl import (
    glEnable,
    glBlendFunc,
    glClearColor,
    GL_BLEND,
    GL_SRC_ALPHA,
    GL_ONE_MINUS_SRC_ALPHA,
)


RAD2DEG = 57.29577951308232
# # Define some colors
_BLACK = (0, 0, 0)
_WHITE = (255, 255, 255)
_RED = (255, 0, 0)
_DARKORANGE = (255, 140, 0)
_TEAL = (0, 128, 128)
_GREY = (60, 60, 60)
_LIGHT_GREY = (200, 200, 200)
_DARK_GREY = (100, 100, 100)

_DARKGREEN = (0, 100, 0)
_NAVY = (0, 0, 128)
_OLIVE = (128, 128, 0)
_MAROON = (128, 0, 0)
_DARKSLATEBLUE = (72, 61, 139)
_PURPLE = (128, 0, 128)


_BACKGROUND_COLOR = _WHITE
_GRID_COLOR = _BLACK

_SHELF_COLORS = [_TEAL, _DARKSLATEBLUE, _PURPLE, _DARKGREEN, _NAVY, _OLIVE, _MAROON]
_SHELF_UNREQUESTED_COLOR = _DARK_GREY
_AGENT_COLOR = _DARKORANGE
_AGENT_LOADED_COLOR = _RED
_GOAL_COLOR = _GREY

_SHELF_PADDING = 6
_AGENT_SIZE = 1 / 2


class Viewer:
    def __init__(self, world_size: tuple[int, int]):
        display = pyglet.display.get_display()
        self.cols, self.rows = world_size

        self.grid_size = 32
        self.icon_size = 20

        self.width = 1 + self.cols * (self.grid_size + 1)
        self.height = 2 + self.rows * (self.grid_size + 1)
        self.window = Window(width=self.width, height=self.height, display=display)  # type: ignore
        self.window.on_close = self.window_closed_by_user  # type: ignore
        self.isopen = True

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def close(self):
        self.window.close()

    def window_closed_by_user(self):
        self.isopen = False
        exit()

    def render(self, env: Warehouse, return_rgb_array: bool = False):
        glClearColor(*_BACKGROUND_COLOR, 0)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()

        batch = Batch()
        grid = self._draw_grid(batch)
        layout = self._draw_layout(env, batch)
        shelves = self._draw_shelves(env, batch)
        agents = self._draw_agents(env, batch)
        batch.draw()

        if return_rgb_array:
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            arr = np.frombuffer(image_data.get_data(), dtype=np.uint8)
            arr = arr.reshape(buffer.height, buffer.width, 4)
            arr = arr[::-1, :, 0:3]
        self.window.flip()

        return arr if return_rgb_array else self.isopen

    def _draw_grid(self, batch: Batch):
        objects = []
        for r in range(self.rows + 1):
            line = Line(
                x=0,
                y=(self.grid_size + 1) * r + 1,
                x2=(self.grid_size + 1) * self.cols,
                y2=(self.grid_size + 1) * r + 1,
                thickness=2,
                color=_GRID_COLOR,
                batch=batch,
            )
            objects.append(line)

        for c in range(self.cols + 1):
            line = Line(
                x=(self.grid_size + 1) * c + 1,
                y=0,
                x2=(self.grid_size + 1) * c + 1,
                y2=(self.grid_size + 1) * self.rows,
                thickness=2,
                color=_GRID_COLOR,
                batch=batch,
            )
            objects.append(line)
        return objects

    def _draw_shelves(self, env: Warehouse, batch: Batch):
        objects = []
        for shelf in env.shelves:
            x, y = shelf.x, shelf.y
            y = self.rows - y - 1  # pyglet rendering is reversed
            if shelf not in env.request_queue:
                shelf_color = _SHELF_UNREQUESTED_COLOR
            else:
                shelf_color = _SHELF_COLORS[shelf.color]

            rect = self._draw_rectangle(
                x, y, color=shelf_color, padding=_SHELF_PADDING, batch=batch
            )
            objects.append(rect)
            if shelf in env.request_queue:
                label = self._draw_char(x, y, "+", color=_WHITE, batch=batch)
                objects.append(label)

        return objects

    def _draw_layout(self, env: Warehouse, batch: Batch):
        objects = []
        # draw highways
        for x in range(self.cols):
            for y_ in range(self.rows):
                should_draw = not env.layout.is_highway(Point(x, y_))
                if should_draw:
                    y = self.rows - y_ - 1  # pyglet rendering is reversed
                    color = _SHELF_COLORS[env.layout._shelf_colors[x, y_]]
                    rect = self._draw_rectangle(
                        x, y, padding=1, color=(*color, 128), batch=batch
                    )
                    objects.append(rect)

        # draw goal boxes
        for goal in env.goals:
            color = _SHELF_COLORS[goal.color]
            x, y = goal.pos
            y = self.rows - y - 1  # pyglet rendering is reversed
            rect = self._draw_rectangle(x, y, color=color, batch=batch)
            label = self._draw_char(x, y, "G", color=_WHITE, batch=batch)
            objects.append(rect)
            objects.append(label)

        return objects

    def _draw_agents(self, env: Warehouse, batch: Batch):
        objects = []
        agent_size = self.grid_size * _AGENT_SIZE
        for agent in env.agents:
            col, row = agent.x, agent.y
            row = self.rows - row - 1

            cx: float = (self.grid_size + 1) * col + self.grid_size // 2 + 1
            cy: float = (self.grid_size + 1) * row + self.grid_size // 2 + 1
            if agent.dir.value == Direction.UP.value:
                x, y = cx, cy + agent_size / 2
                x2, y2 = cx + agent_size / 2, cy - agent_size / 2
                x3, y3 = cx - agent_size / 2, cy - agent_size / 2
            elif agent.dir.value == Direction.RIGHT.value:
                x, y = cx + agent_size / 2, cy
                x2, y2 = cx - agent_size / 2, cy + agent_size / 2
                x3, y3 = cx - agent_size / 2, cy - agent_size / 2
            elif agent.dir.value == Direction.DOWN.value:
                x, y = cx, cy - agent_size / 2
                x2, y2 = cx + agent_size / 2, cy + agent_size / 2
                x3, y3 = cx - agent_size / 2, cy + agent_size / 2
            elif agent.dir.value == Direction.LEFT.value:
                x, y = cx - agent_size / 2, cy
                x2, y2 = cx + agent_size / 2, cy + agent_size / 2
                x3, y3 = cx + agent_size / 2, cy - agent_size / 2

            draw_color = _AGENT_LOADED_COLOR if agent.carried_shelf else _AGENT_COLOR

            triangle = Triangle(
                x=x, y=y, x2=x2, y2=y2, x3=x3, y3=y3, color=draw_color, batch=batch
            )
            objects.append(triangle)
        return objects

    def _draw_rectangle(self, x, y, color, padding: int = 2, batch=None):
        return Rectangle(
            x=(self.grid_size + 1) * x + padding + 1,
            y=(self.grid_size + 1) * y + padding + 1,
            width=self.grid_size - 2 * padding + 1,
            height=self.grid_size - 2 * padding + 1,
            color=color,
            batch=batch,
        )

    def _draw_char(self, x, y, char, color, batch=None):
        label_x = 1 + x * (self.grid_size + 1) + (1 / 2) * (self.grid_size + 1)
        label_y = (self.grid_size + 1) * y + (1 / 2) * (self.grid_size + 1) + 4
        label = pyglet.text.Label(
            char,
            font_name="Monospace",
            font_size=18,
            x=label_x,
            y=label_y,
            anchor_x="center",
            anchor_y="center",
            color=(*color, 255),
            batch=batch,
        )
        return label
