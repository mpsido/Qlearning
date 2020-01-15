import numpy as np
import random

class Game:
    def step(self, action):
        #return state, reward, done, info
        pass

    def render(self):
        pass

    def reset(self):
        pass

    def close(self):
        pass

    # For with statement
    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()

class TreasureCave(Game):

    def __init__(self, verbose=False):
        self.treasure_map=np.array([
            [".",".",".","."],
            [".","0",".","0"],
            [".",".",".","0"],
            ["0",".",".","T"]
        ])
        self.verbose=verbose
        self.reset()
        class ActionSpace:
            def sample(self):
                return random.randrange(self.n)
        self.action_space = ActionSpace()
        self.action_space.n = 4

    def reset(self):
        self.done=False
        self.reward=0
        self.hero_pos=0
        self.evil_pos=14

    @staticmethod
    def get_pos_from_coord(coord):
        return coord[0]*4+coord[1]

    @staticmethod
    def get_coord_from_pos(pos):
        return (int(pos/4), pos%4)

    def is_treasure(self, coord):
        return self.treasure_map[coord[0]][coord[1]] == "T"

    def is_hole(self, coord):
        return self.treasure_map[coord[0]][coord[1]] == "0"

    def is_evil(self, coord):
        return coord == TreasureCave.get_coord_from_pos(self.evil_pos)

    def render(self):
        render_map=self.treasure_map.copy()
        hero_coord=TreasureCave.get_coord_from_pos(self.hero_pos)
        evil_coord=TreasureCave.get_coord_from_pos(self.evil_pos)
        if not self.is_hole(hero_coord):
            render_map[hero_coord[0]][hero_coord[1]]="X"
        if render_map[evil_coord[0]][evil_coord[1]] == "X":
            render_map[evil_coord[0]][evil_coord[1]]="!"
        else:
            render_map[evil_coord[0]][evil_coord[1]]="E"
        print("\n".join([ ''.join(r) for r in render_map]))

    def evil_move(self, hero_coord):
        evil_coord=TreasureCave.get_coord_from_pos(self.evil_pos)
        for a in range(self.action_space.n):
            if self.make_move(evil_coord, a) == hero_coord:
                return a
        return self.action_space.sample()

    def make_move(self, coord, action):
        possible_actions=['w', 's', 'a', 'd']
        action=possible_actions[action]
        if action == 'w':
            if coord[0] > 0:
                coord=(coord[0]-1, coord[1])
        elif action == 's':
            if coord[0] < len(self.treasure_map)-1:
                coord=(coord[0]+1, coord[1])
        elif action == 'a':
            if coord[1] > 0:
                coord=(coord[0], coord[1]-1)
        elif action == 'd':
            if coord[1] < len(self.treasure_map[coord[0]])-1:
                coord=(coord[0], coord[1]+1)
        else:
            raise IndexError(action, "does not exist. Choose action among: a,s,d,w")
        return coord

    def step(self, action):
        if self.done:
            return (self.hero_pos, self.evil_pos), self.reward, self.done, "Game over"
        hero_coord=TreasureCave.get_coord_from_pos(self.hero_pos)
        evil_coord=TreasureCave.get_coord_from_pos(self.evil_pos)
        hero_coord=self.make_move(hero_coord, action)
        evil_coord=self.make_move(evil_coord, self.evil_move(hero_coord))
        self.hero_pos=TreasureCave.get_pos_from_coord(hero_coord)
        self.evil_pos=TreasureCave.get_pos_from_coord(evil_coord)
        if self.is_hole(hero_coord) or self.is_evil(hero_coord):
            self.done=True
            self.reward=-1
        elif self.is_treasure(hero_coord):
            self.done=True
            self.reward=1
        return (self.hero_pos, self.evil_pos), self.reward, self.done, ""