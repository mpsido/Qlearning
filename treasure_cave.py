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
            ['.','.','.','.','.','.','.'],
            ['.','.','0','0','.','0','.'],
            ['0','.','.','0','.','.','.'],
            ['.','.','.','.','.','0','.'],
            ['0','.','.','.','.','T','.']
        ])
        self.possible_actions=['w', 's', 'a', 'd','e']
        self.verbose=verbose
        self.reset()
        class ActionSpace:
            def sample(self):
                return random.randrange(self.n)
        self.action_space = ActionSpace()
        self.action_space.n = len(self.possible_actions)

    def reset(self):
        self.done=False
        self.reward=0
        self.hero_pos=0
        self.evil_pos=[24, 15]
        return (self.hero_pos, self.evil_pos), self.reward, self.done, "New game"

    def get_pos_from_coord(self, coord):
        return coord[0]*self.treasure_map.shape[1]+coord[1]

    def get_coord_from_pos(self, pos):
        x=int(pos/self.treasure_map.shape[1])
        y=pos%self.treasure_map.shape[1]
        return (x, y)

    def is_treasure(self, coord):
        return self.treasure_map[coord[0]][coord[1]] == 'T'

    def is_hole(self, coord):
        return self.treasure_map[coord[0]][coord[1]] == '0'

    def is_evil(self, coord):
        isEvil=False
        for evil_pos in self.evil_pos:
            isEvil |= coord == self.get_coord_from_pos(evil_pos)
        return isEvil

    def render(self):
        render_map=self.treasure_map.copy()
        hero_coord=self.get_coord_from_pos(self.hero_pos)
        if not self.is_hole(hero_coord):
            render_map[hero_coord[0]][hero_coord[1]]='X'
        for evil_pos in self.evil_pos:
            evil_coord=self.get_coord_from_pos(evil_pos)
            if render_map[evil_coord[0]][evil_coord[1]] == 'X':
                render_map[evil_coord[0]][evil_coord[1]]='!'
            else:
                render_map[evil_coord[0]][evil_coord[1]]='E'
        print('\n'.join([ ''.join(r) for r in render_map]))

    def evil_move(self, hero_coord, evil_coord):
        for a in self.possible_actions:
            if self.make_move(evil_coord, a) == hero_coord:
                return a
        return random.choice(self.possible_actions)

    def make_move(self, coord, action):
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
        elif action == 'e':
            pass
        else:
            raise IndexError(action, "does not exist. Choose action among: ", self.possible_actions)
        return coord

    def step(self, action):
        action=self.possible_actions[action]
        return self.play_step(action)

    def play_step(self, action):
        if self.done:
            return (self.hero_pos, self.evil_pos), self.reward, self.done, "Game over"
        hero_coord=self.get_coord_from_pos(self.hero_pos)
        hero_coord=self.make_move(hero_coord, action)
        self.hero_pos=self.get_pos_from_coord(hero_coord)
        if self.is_hole(hero_coord):
            self.done=True
            self.reward=-1
        elif self.is_treasure(hero_coord):
            self.done=True
            self.reward=1
        else:
            # todo update evil coord in state
            for i,evil_pos in enumerate(self.evil_pos):
                evil_coord=self.get_coord_from_pos(evil_pos)
                evil_coord=self.make_move(evil_coord, self.evil_move(hero_coord, evil_coord))
                self.evil_pos[i]=self.get_pos_from_coord(evil_coord)
                if self.is_evil(hero_coord):
                    self.done=True
                    self.reward=-1
        return (self.hero_pos, self.evil_pos), self.reward, self.done, ""
