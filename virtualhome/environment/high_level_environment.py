import numpy as np
import random

ROOM = ['kitchen', 'bedroom', 'livingroom', 'bathroom']
objs = ['alcohol', 'amplifier', 'apple', 'bananas', 'barsoap', 'bathtub', 'bed', 'bellpepper', 'bench', 'boardgame', 'book', 'bottlewater',
        'breadslice', 'carrot', 'ceilingfan', 'cellphone', 'cereal', 'chair', 'chicken', 'chips', 'chocolatesyrup', 'clock', 'closet', 'coffeemaker',
        'coffeepot', 'computer', 'condimentbottle', 'condimentshaker', 'cookingpot', 'cpuscreen', 'crayons', 'creamybuns', 'cupcake', 'curtains',
        'cutleryfork', 'cutleryknife', 'cutlets', 'cuttingboard', 'deodorant', 'desk', 'dishbowl', 'dishwasher', 'dishwashingliquid', 'doorjamb',
        'faucet', 'folder', 'fridge', 'fryingpan', 'garbagecan', 'hairproduct', 'hanger', 'juice', 'keyboard', 'knifeblock', 'lightswitch',
        'lime', 'magazine', 'microwave', 'milk', 'milkshake', 'mincedmeat', 'mouse', 'mousemat', 'mug', 'nightstand', 'notes', 'painkillers',
        'pancake', 'paper', 'papertray', 'peach', 'pear', 'perfume', 'photoframe', 'pie', 'plate', 'plum', 'poundcake', 'powersocket', 'printer',
        'pudding', 'radio', 'remotecontrol', 'salad', 'salmon', 'sink', 'sofa', 'speaker', 'stove', 'stovefan', 'tablelamp', 'toaster', 'toiletpaper',
        'toothbrush', 'toothpaste', 'towel', 'toy', 'tv', 'tvstand', 'washingmachine', 'washingsponge', 'waterglass', 'whippedcream', 'wine', 'wineglass']

class ExpertPolicy(object):
    def __init__(self):
        self.room = None
        self.high_leve_script = None
        self.temp_instruction_idx = 0
        self.room_structure = None
        self.available_objects = None

        self.rotate = False
        self.rotate_action = "turnright"
        self.rotation_count = 0
        self.rotate_objs_num = []

    def reset(self, script):
        self.high_level_script = script
        self.temp_instruction_idx = 0
        self.room_finding = 0
        self.room = "kitchen"

    def set_available_objects(self, available_objects):
        self.available_objects = available_objects

    def step(self, visible_graph, agent_graph):
        if len(self.high_level_script) <= self.temp_instruction_idx:
            return None
        rooms = ["kitchen", "livingroom", "bathroom", "bedroom"]
        for edge in agent_graph['edges']:
            if edge[1] == 'INSIDE':
                agent_room = edge[2]
        temp_instruction = self.high_level_script[self.temp_instruction_idx]
        temp_inst_token_list = temp_instruction.split()
        target_action, target_object = temp_inst_token_list[0], temp_inst_token_list[1]
        action = None

        if self.type_of_instruction(temp_instruction) == "walk_object":
            if target_object in self.available_objects:
                self.temp_instruction_idx += 1
                action = "walk {}".format(target_object)
            elif self.rotation_count >= 3 and len(visible_graph['nodes']) >= np.mean(self.rotate_objs_num) - self.rotation_count:
                action = "walkforward"
        elif self.type_of_instruction(temp_instruction) == "walk_room":
            if target_object in self.available_objects:
                self.temp_instruction_idx += 1
                self.room_finding = 0
                action = "walk {}".format(target_object)
            elif self.rotation_count >= 6 - self.room_finding and len(visible_graph['nodes']) >= np.mean(self.rotate_objs_num) - 1:
                self.room_finding = 3
                action = "walkforward"
            elif self.rotation_count >= 3 - self.room_finding:
                for room in rooms:
                    if room in self.available_objects and room != agent_room:
                        self.room_finding = 0
                        action = "walk {}".format(room)
        elif self.type_of_instruction(temp_instruction) == "interaction":
            if target_object in self.available_objects:
                self.temp_instruction_idx += 1
                action = "{} {}".format(target_action, target_object)

        if action is None:
            self.rotate_objs_num.append(len(visible_graph['nodes']))
            return self.get_rotate_action()
        else:
            self.rotate = False
            self.rotation_count = 0
            return action


    def type_of_instruction(self, instruction):
        inst_list = instruction.split()
        rooms = ["kitchen", "livingroom", "bathroom", "bedroom"]
        if inst_list[0] == "walk":
            if inst_list[1] in rooms:
                return "walk_room"
            else:
                return "walk_object"
        else:
            return "interaction"

    def get_rotate_action(self):
        if not self.rotate:
            self.rotate_objs_num = []
            self.rotate_action = random.choice(["turnright", "turnleft"])
            self.rotate = True
        self.rotation_count += 1
        return self.rotate_action

    def get_near_room(self, room=None):
        if room is None:
            return self.room_structure[self.room]
        else:
            return self.room_structure[room]

    def get_next_room(self, target_room):
        if target_room in self.get_near_room():
            return target_room
        else:
            for room in self.get_near_room():
                if target_room in self.get_near_room(room):
                    return room
        return None

class HighLevelEnv(object):
    def __init__(self, env):
        self.timestep = 0
        self.max_timestep = 15
        self.env = env
        self.policy = ExpertPolicy()
        self.action_space = [
            "navigate",
            "go",
            "grab",
            # "switch on",
            # "examine",
            # "plug in",
            "put",
        ]

    def step(self, action):
        self.timestep += 1
        split_action = action.split()
        # if split_action[0] == 'navigate' and len(split_action) == 4:
        #     self.policy.reset([f"walk {split_action[3]}", f"walk {split_action[1]}"])
        #     for _ in range(20):
        #         avail_action, name_to_id = self.env.actions_available()
        #         self.policy.set_available_objects(name_to_id)
        #         command = self.policy.step(self.env.get_visible_graph(mode='triples'), self.env.get_agent_graph(mode='triples'))
        #         if command is None:
        #             break
        #         obs, reward, done, info = self.env.step(command)
        if split_action[0] == 'navigate':
            self.policy.reset([f"walk {split_action[1]}"])
            for trial in range(20):
                avail_action, name_to_id = self.env.actions_available()
                self.policy.set_available_objects(name_to_id)
                command = self.policy.step(self.env.get_visible_graph(mode='triples'), self.env.get_agent_graph(mode='triples'))
                if command is None:
                    break
                obs, reward, done, info = self.env.step(command)
                if trial == 19:
                    info['navigate_failed'] = True
        elif split_action[0] == 'go':
            self.policy.reset([f"walk {split_action[1]}"])
            for trial in range(20):
                avail_action, name_to_id = self.env.actions_available()
                self.policy.set_available_objects(name_to_id)
                command = self.policy.step(self.env.get_visible_graph(mode='triples'), self.env.get_agent_graph(mode='triples'))
                if command is None:
                    break
                obs, reward, done, info = self.env.step(command)
                if trial == 19:
                    info['navigate_failed'] = True
        else:
            obs, reward, done, info = self.env.step(action)

        if self.max_timestep == self.timestep:
            done = True
        if "navigate_failed" not in info.keys():
            info["navigate_failed"] = False
        return obs, reward, done, info

    def reset(self, environment_id=0):
        self.timestep = 0
        return self.env.reset(environment_id=environment_id)

    def available_scripts(self):
        avail_action, name_to_id = self.env.actions_available()
        graph = self.env.get_graph(mode='triples')
        agent_graph = self.env.get_agent_graph(mode='triples')
        available_scripts = []
        avail_action.append('navigate')
        for act in self.action_space:
            if act == 'put':
                for edge in agent_graph['edges']:
                    if edge[1].lower() == "hold":
                        for obj_edge in graph['edges']:
                            if obj_edge[1].lower() == "close" and obj_edge[2] != 'character':
                                available_scripts.append(' '.join([act, edge[2], obj_edge[2]]))
            elif act == 'grab':
                avail_objects = name_to_id.keys()
                for obj in avail_objects:
                    available_scripts.append(' '.join([act, obj]))
            elif act == 'go':
                for room in ROOM:
                    available_scripts.append(f"go {room}")
            else:
                for obj in name_to_id.keys():
                    if obj != 'character' and obj not in ROOM:
                        available_scripts.append(f"navigate {obj}")
        return available_scripts


    def programming_description(self, examples):
        import_action_primitives = "from actions import navigate <obj>, go <room>, grab <object>, switchon <object>, put <obj> <obj>"
        available_objects_list = "objects = [{}]\nrooms=[{}]".format(", ".join(objs), ", ".join(ROOM))

