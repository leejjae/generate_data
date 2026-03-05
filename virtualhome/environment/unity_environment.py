import sys
import os

curr_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{curr_dir}/../')

from collections import defaultdict
from unity_simulator import comm_unity as comm_unity
from evolving_graph import utils

from .utils import REL_EQUAL
from .resources import *

import atexit
import pdb
import ipdb
import random
import cv2
import numpy as np

class UnityEnvironment:
    def __init__(
        self,
        num_agents=2,
        max_episode_length=500,
        observation_types=None,
        use_editor=False,
        base_port=8080,
        port_id=0,
        executable_args=None,
        recording_options=None,
        seed=123,
        url='127.0.0.1'
    ):

        # Random seed
        self.seed = seed
        self.rnd = random.Random(seed)
        np.random.seed(seed)

        # Recording option
        if recording_options is None:
            self.recording_options = {'recording': False,
                                    'output_folder': None,
                                    'file_name_prefix': None,
                                    'cameras': 'PERSON_FROM_BACK',
                                    'modality': 'normal'}
        else:
            self.recording_options = recording_options

        # Observation parameters
        self.num_camera_per_agent = 6
        self.CAMERA_NUM = [1, 2] # 0 TOP, 1 FRONT, 2 LEFT..
        self.default_image_width = 600
        self.default_image_height = 250

        if observation_types is not None:
            self.observation_types = observation_types
        else:
            self.observation_types = ['partial' for _ in range(num_agents)]

        self.agent_info = {
            0: 'Chars/Male1',
            1: 'Chars/Female1'
        }

        self.timesteps = 0
        self.env_id = 0
        self.prev_reward = 0
        self.num_agents = num_agents
        self.max_episode_length = max_episode_length

        self.max_ids = {}
        self.changed_graph = True
        self.changed_visible_object = True
        self.changed_image_view = True
        self.rooms = None
        self.id2node = None
        self.num_static_cameras = None
        self.success_condition = None
        self.rgb_to_instance = None
        self.prev_observable_objects = {}
        self.visible_objects_cache = {}
        self.image_cache = None
        self.rooms2id = {}
        self.temp_manipulated_object = None
        self.switchable_object = []
        self.openable_object = []

        self.base_port = base_port
        self.port_id = port_id
        self.executable_args = {} if executable_args is None else executable_args
        self.room_adjacent = room_adjacent
        self.valid_env_id = list(self.room_adjacent.keys())
        if use_editor:
            # Use Unity Editor
            self.port_number = 8080
            self.comm = comm_unity.UnityCommunication(url=url,)
        else:
            # Launch the executable
            self.port_number = self.base_port + port_id
            # ipdb.set_trace()
            self.comm = comm_unity.UnityCommunication(url=url,
                                                      port=str(self.port_number),
                                                      **self.executable_args)

        atexit.register(self.close)
        # self.reset()

    def close(self):
        self.comm.close()

    def relaunch(self):
        self.comm.close()
        self.comm = comm_unity.UnityCommunication(port=str(self.port_number), **self.executable_args)

    def set_task(self, success_condition):
        """
        :param success_condition: List[Tuple[str]]
        success condition for graph edges
        e.g. if success_condition is {"required_condition": [("tv", "is", "on")], "prohibited_condition": [("agent", "inside", "bathroom")]},
        the task is success when ("tv", "is", "on") is in environment graph and ("agent", "inside", "bathroom") is not in environment graph.
        :return: None
        """
        self.success_condition = success_condition

    def reward(self):
        """
        Calculate reward for task
        :return: reward, done, info
        """
        if self.success_condition is None:
            return 0, False, {}

        info = dict()
        reward = 0

        success = [False] * len(self.success_condition["required_condition"])
        entire_graph = self.get_graph(mode='triples')
        agent_graph = self.get_agent_graph(mode='triples')

        for edge in entire_graph['edges']:
            for idx, suc_cond in enumerate(self.success_condition["required_condition"]):
                if REL_EQUAL(edge, suc_cond):
                    success[idx] = True

        for edge in agent_graph['edges']:
            for idx, suc_cond in enumerate(self.success_condition["required_condition"]):
                if REL_EQUAL(edge, suc_cond):
                    success[idx] = True


        for i in range(len(success)):
            if success[i]:
                reward += 1
        if reward == len(success):
            info['is_success'] = True
            done = True
        else:
            info['is_success'] = False
            done = False
        info['condwise_success'] = success
        info['success_condition'] = self.success_condition
        return reward, done, info

    def step(self, action_dict):
        if action_dict == "look around":
            obs_seq = []
            for i in range(4):
                obs, reward, done, info = self._step("turnleft")
                obs_seq.append(obs)
            return obs_seq, reward, done, info
        else:
            return self._step(action_dict)

    def _step(self, action_dict):
        prev_reward, _, _ = self.reward()
        avail_actions, avail_objects = self.actions_available(action_dict.split()[0])
        script_list = self.convert_action(action_dict, avail_objects)
        self.temp_manipulated_object = []
        if len(script_list[0]) > 0:
            if self.recording_options['recording'] and "no_action" not in script_list[0]:
                success, message = self.comm.render_script(script_list,
                                                           recording=True,
                                                           skip_animation=False,
                                                           camera_mode=self.recording_options['cameras'],
                                                           file_name_prefix='recording',
                                                           image_synthesis=self.recording_options['modality'])
            else:
                success, message = self.comm.render_script(script_list,
                                                           recording=False,
                                                           skip_animation=True)
            if not success:
                pass
                #print(message)
            else:
                if len(action_dict.split()) > 1:
                    if action_dict.split()[1] not in self.rooms:
                        self.temp_manipulated_object = action_dict.split()[1:]
                self.changed_graph = True
                self.changed_visible_object = True
                self.changed_image_view = True
        # Obtain reward
        reward, done, info = self.reward()
        # if self.success_condition[0][1] == 'put' and action_dict.split()[0] == 'put' and script_list[0] != '<char0> [no_action]':
        #     if self.success_condition[0][0] == action_dict.split()[1] and self.success_condition[0][2] == action_dict.split()[2]:
        #         info['is_success'] = True
        #         done = True
        self.timesteps += 1

        obs = self.get_observations()
        info['finished'] = done
        info['success'] = success
        info['earn_reward'] = reward - prev_reward

        self.prev_observable_objects = avail_objects
        if self.timesteps == self.max_episode_length:
            done = True
        return obs, reward, done, info


    def reset(self, environment_graph=None, environment_id=None, init_rooms=None):
        """
        :param environment_graph: the initial graph we should reset the environment with
        :param environment_id: which id to start
        :param init_rooms: where to intialize the agents
        """
        self.env_id = environment_id
        self._initialize_environment()

        if environment_graph is None:
            updated_graph = self._generate_initial_graph()
        else:
            updated_graph = environment_graph

        success, message = self.comm.expand_scene(updated_graph)
        if not success:
            print("Error expanding scene")
            pdb.set_trace()
            return None

        self._initialize_cameras()
        self._initialize_rooms(init_rooms)
        self._initialize_object_information()
        # self._initialize_task_relevant_objects()
        self._check_valid_environment()

        obs = self.get_observations()
        return obs

    def _initialize_environment(self):
        """Initialize the environment by resetting and setting the max IDs for new objects."""
        if self.env_id is not None:
            self.comm.reset(self.env_id)
        else:
            self.comm.reset()


        _, graph = self.comm.environment_graph()
        if self.env_id not in self.max_ids:
            self.max_ids[self.env_id] = max(node['id'] for node in graph['nodes'])

        self.timesteps = 0
        self.prev_reward = 0.
        self.changed_graph = self.changed_visible_object = self.changed_image_view = True

    def _generate_initial_graph(self):
        """Generate the initial environment graph based on predefined object classes."""
        entire_graph = self.get_graph(reset_adjacent=False)
        updated_graph = {'nodes': [], 'edges': []}
        classwise_obj = []
        node_ids = []
        for node in entire_graph['nodes']:
            if node['class_name'].lower() in [item for tup in self.success_condition['required_condition'] for item in tup]:
                if node['class_name'] not in classwise_obj:
                    classwise_obj.append(node['class_name'])
                    updated_graph['nodes'].append(node)
                    node_ids.append(node['id'])
            else:
                updated_graph['nodes'].append(node)
                node_ids.append(node['id'])

        for edge in entire_graph['edges']:
            if edge['from_id'] in node_ids and edge['to_id'] in node_ids:
                updated_graph['edges'].append(edge)

        return updated_graph

    def _initialize_cameras(self):
        """Set up the environment cameras."""
        self.num_static_cameras = self.comm.camera_count()[1]
        camera_positions = [[0, 0.5, 0], [0, 1.5, 0]]
        for i, pos in enumerate(camera_positions):
            self.comm.update_camera(self.num_static_cameras + self.CAMERA_NUM[i], field_view=100, position=pos, rotation=[0, 0, 0])

    def _initialize_rooms(self, init_rooms):
        """Initialize rooms based on provided or default room names."""
        valid_rooms = ['kitchen', 'bedroom', 'livingroom', 'bathroom']
        if init_rooms is None or init_rooms[0] not in valid_rooms:
            rooms = self.rnd.sample(valid_rooms, 3)
        else:
            rooms = list(init_rooms)

        for i in range(self.num_agents):
            room = rooms[2] if i in self.agent_info else None
            self.comm.add_character(self.agent_info.get(i), initial_room=room)

        self.changed_graph = self.changed_visible_object = self.changed_image_view = True

    def _initialize_task_relevant_objects(self):
        graph = self.get_graph()
        """Identify and store initial positions of task-relevant objects."""
        self.task_relevant_orginal_pos = {}
        _target_obj = ["towel", "paper", "book", "mug", "plate", "apple"]

        target_obj = [cond[0] for cond in self.success_condition if cond[0] in _target_obj]
        for obj in target_obj:
            id_list = [node['id'] for node in graph['nodes'] if node['class_name'].lower() == obj]
            if id_list:
                for edge in graph['edges']:
                    if edge['from_id'] == id_list[0] and edge['relation_type'] == "ON":
                        self.task_relevant_orginal_pos[obj] = self.id2node[edge['to_id']]['class_name']

        self.task_relevant_orginal_pos["book"] = "tvstand"  # Override for "book"
        self.task_relevant_movable_obj = list(self.task_relevant_orginal_pos.keys())
        self.reset_object_pos()

    def _initialize_object_information(self):
        graph = self.get_graph()
        self.temp_manipulated_object = []
        self.prev_observable_objects = {}
        self.rooms = {node['id']: node['class_name'] for node in graph['nodes'] if node['category'] == 'Rooms'}
        self.id2node = {node['id']: node for node in graph['nodes']}
        self.objinroom = {edge['from_id']: edge['to_id'] for edge in graph['edges'] if edge['relation_type'] == 'INSIDE' and edge['from_id'] != 1 and edge['to_id'] in self.rooms}
        self.rgb_to_instance = self.get_rgb_to_instance()

    def _check_valid_environment(self):
        graph = self.get_graph()
        name_to_ids  = [node['class_name'].lower() for node in graph['nodes']]

        for success_condition in self.success_condition['required_condition']:
            source_entity, relation, target_entity = success_condition
            if source_entity.lower() not in name_to_ids:
                raise NotImplementedError()
            elif relation.lower() != "is" and target_entity not in name_to_ids:
                raise NotImplementedError()

    def reset_object_pos(self, action="place", obj="all"):
        if action == "place":
            hold_obj = self.get_hold_objects()
            if obj == "all":
                for i in range(len(self.task_relevant_movable_obj)):
                    if not hold_obj or (hold_obj and self.id2node[hold_obj[0]]['class_name'] != self.task_relevant_movable_obj[i]):
                        self.command("grab", self.task_relevant_movable_obj[i])
                        self.command("put", self.task_relevant_orginal_pos[self.task_relevant_movable_obj[i]])
            else:
                if obj in self.task_relevant_movable_obj:
                    if not hold_obj or (hold_obj and self.id2node[hold_obj[0]]['class_name'] != obj):
                        self.command("grab", obj)
                        self.command("put", self.task_relevant_orginal_pos[obj])
                    return True
                else:
                    return False
        else:
            self.command(action, obj)
            return True

    def convert_action(self, action, avail_objects):
        # avail_objects.update(self.prev_observable_objects)
        action_list = action.split()

        if action_list[0] == "switch":
            action_list[0] = "switchon"
        elif action_list[0] == "plug":
            action_list[0] = "plugin"

        try:
            if len(action_list) == 1:
                current_script = ['<char{}> [{}]'.format(0, action)]
                if action_list[0] == 'turnright' or action_list[0] == 'turnleft':
                    current_script.append('<char{}> [{}]'.format(0, action))
                    current_script.append('<char{}> [{}]'.format(0, action))
                elif action_list[0] == 'walkforward':
                    current_script.append('<char{}> [{}]'.format(0, action))
            elif len(action_list) == 2:
                if action_list[1] in avail_objects.keys() and action_list[0] == "put" or action_list[0] == "putin":
                    hold_object = self.get_hold_objects()
                    if hold_object and action_list[1] in avail_objects.keys():
                        current_script = ['<char{}> [{}] <{}> ({}) <{}> ({})'.format(0, action_list[0], self.id2node[hold_object[0]]['class_name'], hold_object[0], action_list[1], avail_objects[action_list[1]])]
                    else:
                        current_script = ['<char0> [no_action]']
                elif action_list[1] in avail_objects.keys():
                    current_script = ['<char{}> [{}] <{}> ({})'.format(0, action_list[0], action_list[1], avail_objects[action_list[1]])]
                else:
                    current_script = ['<char0> [no_action]']
            elif len(action_list) == 3:
                if action_list[1] in avail_objects.keys():
                    current_script = ['<char{}> [{}] <{}> ({}) <{}> ({})'.format(0, action_list[0], action_list[1], avail_objects[action_list[1]], action_list[2], avail_objects[action_list[2]])]
                else:
                    current_script = ['<char0> [no_action]']
            else:
                current_script = ['<char0> [no_action]']
        except Exception as e:
            current_script = ['<char0> [no_action]']
            print(e)

        return current_script

    def command(self, action, obj):
        graph = self.get_graph()
        id_list = []
        obj_room = []
        for node in graph['nodes']:
            if node['class_name'].lower() == obj:
                id_list.append(node['id'])

        for edge in graph['edges']:
            for from_id in id_list:
                if edge['from_id'] == from_id and edge['relation_type'] == "INSIDE" and edge['to_id'] in self.rooms:
                    obj_room.append((self.id2node[edge['to_id']]['class_name'], edge['to_id']))

        script_list = []
        for idx, obj_id in enumerate(id_list):
            # script_list.append("<char1> [walk] <{}> ({})".format(obj_room[idx][0], obj_room[idx][1]))
            script_list.append("<char1> [walk] <{}> ({})".format(obj, obj_id))
            hold_object = self.get_hold_objects(character=2)
            if action == 'put':
                script_list.append('<char1> [put] <{}> ({}) <{}> ({})'.format(self.id2node[hold_object[0]]['class_name'], hold_object[0], obj, obj_id))
            else:
                script_list.append("<char1> [{}] <{}> ({})".format(action, obj, obj_id))
            # script_list.append("<char1> [{}] <{}> ({})".format(action, obj, obj_id))
            break

        if not script_list:
            return

        # print(script_list)
        for script in script_list:
            if self.recording_options['recording']:
                success, message = self.comm.render_script([script],
                                                           recording=True,
                                                           skip_animation=False,
                                                           camera_mode=self.recording_options['cameras'],
                                                           file_name_prefix=self.recording_options['file_name_prefix'],
                                                           image_synthesis=self.recording_options['modality'])
            else:
                success, message = self.comm.render_script([script],
                                                           recording=False,
                                                           skip_animation=True)

        # print(success, message)
        if not success:
            print(message)
        else:
            # print(message)
            self.changed_graph = True
            self.changed_visible_object = True
            self.changed_image_view = True


    def actions_available(self, action):
        # visible_object = self.get_visible_objects()
        # name_to_id = dict()
        # if visible_object[0]:
        #     for key, item in visible_object[1].items():
        #         name_to_id[item] = key

        if action == 'walk':
            same_room_object = self.get_same_room_objects()
            name_to_id = dict()
            if same_room_object[0]:
                for key in same_room_object:
                    name_to_id[self.id2node[key]['class_name']] = key
        else:
            close_object_id = self.get_close_objects()
            name_to_id = dict()
            if close_object_id:
                for key in close_object_id:
                    name_to_id[self.id2node[key]['class_name']] = key

        avail_action = [
            'walkforward',
            'turnleft',
            'turnright',
            'walk',
            'switchon',
            'put',
            'open',
            'close',
            'grab',
        ]
        return avail_action, name_to_id

    def get_graph(self, mode='json', with_properties=False, reset_adjacent=True):
        if self.changed_graph:
            s, graph = self.comm.environment_graph()
            if not s:
                pdb.set_trace()
            self.graph = graph
            self.changed_graph = False

            if reset_adjacent:
                for node in self.graph['nodes']:
                    if node["prefab_name"] == 'Female1':
                        node['class_name'] = "Person"
                self.rooms2id = {node['class_name']: node['id'] for node in self.graph['nodes'] if node['category'] == 'Rooms'}
                for i in range(len(self.room_adjacent[self.env_id])):
                    self.graph["edges"].append({"from_id": self.rooms2id[self.room_adjacent[self.env_id][i][0]],
                                                "to_id": self.rooms2id[self.room_adjacent[self.env_id][i][1]],
                                                "relation_type": "ADJACENT"})
                    self.graph["edges"].append({"from_id": self.rooms2id[self.room_adjacent[self.env_id][i][1]],
                                                "to_id": self.rooms2id[self.room_adjacent[self.env_id][i][0]],
                                                "relation_type": "ADJACENT"})

        if mode == 'json':
            return self.graph
        elif mode == 'triples':
            object_ids = self.get_nodes().keys()
            object_ids = [int(x) for x in object_ids]
            temp_kg = self.graph

            visible_kg = {'nodes': [], 'edges': []}
            id_to_class_name = {}
            for node in temp_kg['nodes']:
                id_to_class_name[node['id']] = node['class_name']
                if node['id'] in object_ids:
                    if node['category'] in ['Walls', 'Ceiling', 'Lamps', 'Floor', 'Decor', 'Floors', 'Doors', 'Windows']:
                        object_ids.remove(node['id'])
                    else:
                        visible_kg['nodes'].append(node['class_name'])
                        if node['states']:
                            for s in node['states']:
                                visible_kg['edges'].append((node['class_name'], "is", s))
                        if with_properties and node['properties']:
                            for p in node['properties']:
                                visible_kg['edges'].append((node['class_name'], "is", p))
            for edge in temp_kg['edges']:
                if edge['from_id'] in object_ids and edge['to_id'] in object_ids and edge['relation_type'] != "FACING":
                    visible_kg['edges'].append((id_to_class_name[edge['from_id']], edge['relation_type'], id_to_class_name[edge['to_id']]))
            return visible_kg
        else:
            raise NotImplementedError()

    def get_position_graph(self, mode='json'):
        temp_kg = self.get_graph()
        position_kg = {'nodes': [], 'edges': []}
        id_to_class_name = {}
        obj_id = []
        for node in temp_kg['nodes']:
            id_to_class_name[node['id']] = node['class_name']
            if node['category'] not in BASE_COMPONENT:
                obj_id.append(node['id'])
                if mode == 'json':
                    position_kg['nodes'].append(node)
                else:
                    position_kg['nodes'].append(node['class_name'].lower())
        for edge in temp_kg['edges']:
            if edge['from_id'] in obj_id and edge['to_id'] in self.rooms and edge['relation_type'] in ["INSIDE", "ON", "ADJACENT"]:
                if mode == 'json':
                    position_kg['edges'].append(edge)
                else:
                    position_kg['edges'].append((id_to_class_name[edge['from_id']], edge['relation_type'], id_to_class_name[edge['to_id']]))
        return position_kg


    def get_nodes(self, entire_nodes=True):
        objects_id_to_name = {}
        entire_graph = self.get_graph()
        for node in entire_graph['nodes']:
            if entire_nodes or node['category'] not in ['Decor', 'Walls', 'Ceiling', 'Lamps', 'Floor', 'Floors', 'Doors', 'Windows']:
                objects_id_to_name[node["id"]] = node['class_name']
        return objects_id_to_name

    def get_visible_graph(self, mode='json'):
        visible_object = self.get_visible_objects()
        if visible_object[0]:
            object_ids = visible_object[1].keys()
            object_ids = [int(x) for x in object_ids]
            temp_kg = self.get_graph()
            # temp_kg = utils.get_visible_nodes(curr_graph, agent_id=(agent_id+1))

            visible_kg = {'nodes': [], 'edges': []}
            id_to_class_name = {}
            for node in temp_kg['nodes']:
                id_to_class_name[node['id']] = node['class_name']
                if node['id'] in object_ids or node['class_name'].lower() in self.temp_manipulated_object:
                    if node['category'] in BASE_COMPONENT and node['id'] in object_ids:
                        object_ids.remove(node['id'])
                    else:
                        if mode == 'json':
                            visible_kg['nodes'].append(node)
                            if node['states']:
                                for s in node['states']:
                                    visible_kg['edges'].append({"from_id": node['id'], "relation_type": "IS", "to_id": s})
                        else:
                            visible_kg['nodes'].append(node['class_name'].lower())
                            if node['states']:
                                for s in node['states']:
                                    visible_kg['edges'].append((node['class_name'], "is", s))
            for edge in temp_kg['edges']:
                if edge['from_id'] in object_ids and edge['to_id'] in object_ids and edge['relation_type'] in ["INSIDE", "ON"]:
                    if mode == 'json':
                        visible_kg['edges'].append(edge)
                    else:
                        visible_kg['edges'].append((id_to_class_name[edge['from_id']], edge['relation_type'], id_to_class_name[edge['to_id']]))
            return visible_kg
        else:
            return None

    def get_agent_graph(self, mode='json'):
        temp_kg = self.get_graph()
        visible_kg = {'nodes': [], 'edges': []}
        visible_object = []
        agent_relation_node = []
        id_to_class_name = self.get_nodes()
        for edge in temp_kg['edges'] :
            if edge['from_id'] == 1 and self.id2node[edge['to_id']]['category'] not in BASE_COMPONENT:
                if mode == 'json' and edge['relation_type'] in ['CLOSE', "HOLDS_RH", "HOLDS_LH", "INSIDE"]:
                    visible_kg['edges'].append(edge)
                    agent_relation_node.append(edge['to_id'])
                elif edge['relation_type'] in ['CLOSE', "HOLDS_RH", "HOLDS_LH", "INSIDE"]:
                    if edge['relation_type'] == 'HOLDS_RH' or edge['relation_type'] == 'HOLDS_LH':
                        rel = 'HOLD'
                    elif edge['relation_type'] == 'CLOSE':
                        continue
                    else:
                        rel = edge['relation_type']
                    visible_kg['edges'].append((id_to_class_name[edge['from_id']], rel, id_to_class_name[edge['to_id']]))
                visible_object.append(edge['to_id'])

        for node in temp_kg['nodes']:
            if node['id'] in visible_object or node['id'] == 1 or node['id'] in agent_relation_node:
                if node['category'] in BASE_COMPONENT:
                    visible_object.remove(node['id'])
                else:
                    if mode == 'json':
                        visible_kg['nodes'].append(node)
                    else:
                        visible_kg['nodes'].append(node['class_name'])
        return visible_kg

    def get_close_objects(self):
        graph = self.get_graph()
        close_object = {edge['from_id']: edge['to_id'] for edge in graph['edges'] if edge['relation_type'] == 'CLOSE' and edge['from_id'] != 1 and edge['to_id'] == 1}
        obj_id_same_room = list(close_object.keys())
        return obj_id_same_room

    def get_hold_objects(self, character=1):
        graph = self.get_graph()
        hold_objects = [edge['to_id'] for edge in graph['edges'] if (edge['relation_type'] == 'HOLDS_RH' or edge['relation_type'] == 'HOLDS_LH') and edge['from_id'] == character]
        return hold_objects

    def get_same_room_objects(self):
        graph = self.get_graph()
        self.objinroom = {edge['from_id']: edge['to_id'] for edge in graph['edges'] if edge['relation_type'] == 'INSIDE' and edge['to_id'] in self.rooms}
        temp_room = self.objinroom[1]
        obj_id_same_room = [key for key, value in self.objinroom.items() if value == temp_room] + [edge["to_id"] for edge in graph['edges'] if edge['relation_type'] == 'ADJACENT' and edge['from_id'] == temp_room]
        return obj_id_same_room

    def get_visible_objects(self):
        """
        :param agent_id:
        :return: Tuple[bool, Dict[str, str]]
        bool -> object existence
        dict -> {object id: object category}
        """
        if not self.changed_visible_object:
            return True, self.visible_objects_cache
        seg_info = {'image_width': 160, 'image_height': 80, "mode": 'seg_inst'}
        image = self.get_observation(agent_id=0, obs_type='image', info=seg_info)

        seen_objects = set()

        pixels = np.array(image)
        pixels = pixels.reshape(-1, pixels.shape[-1])
        unique_colors = np.unique(pixels, axis=0)

        for pixel in unique_colors:
            if tuple(pixel) in self.rgb_to_instance:
                if len(self.rgb_to_instance[tuple(pixel)]) > 1:
                    continue
                instance = self.rgb_to_instance[tuple(pixel)][0]
                seen_objects.add(int(instance))

        vis_dict = self.get_observation(agent_id=0, obs_type='visible', info={'image_type': 'concat'})
        node_info = self.get_nodes(entire_nodes=False)

        for seen_id in seen_objects:
            if int(seen_id) in node_info:
                vis_dict[str(seen_id)] = node_info[int(seen_id)]

        seen_values = set()
        new_vis_dict = {}
        for key, value in vis_dict.items():
            if value not in seen_values:
                seen_values.add(value)
                new_vis_dict[int(key)] = value

        visible_rooms = set()
        for obj_id in new_vis_dict.keys():
            if obj_id in self.objinroom:
                visible_rooms.add(self.objinroom[obj_id])

        for room_id in visible_rooms:
            new_vis_dict[room_id] = self.rooms[room_id]
        self.visible_objects_cache = new_vis_dict
        self.changed_visible_object = False
        return True, new_vis_dict

    def get_rgb_to_instance(self):
        instance_colors = self.comm.instance_colors()
        rgb_to_instance = {}
        for k, v in instance_colors[1].items():
            rgb_code = [round(v[2] * 255), round(v[1] * 255), round(v[0] * 255)]
            for i in range(3):
                if rgb_code[i] == 70:
                    rgb_code[i] = 69
            rgb_code = tuple(rgb_code)
            if tuple(v) not in rgb_to_instance.keys():
                rgb_to_instance[rgb_code] = []
            rgb_to_instance[rgb_code].append(k)
        return rgb_to_instance

    def get_observations(self):
        dict_observations = {}
        dict_observations['image'] = self.get_observation(agent_id=0, obs_type='image')
        dict_observations['visible_graph'] = self.get_visible_graph()
        dict_observations['agent_graph'] = self.get_agent_graph()
        return dict_observations

    def get_observation(self, agent_id, obs_type, info={}):
        if obs_type == 'partial':
            # agent 0 has id (0 + 1)
            curr_graph = self.get_graph()
            return utils.get_visible_nodes(curr_graph, agent_id=(agent_id+1))

        elif obs_type == 'full':
            return self.get_graph()

        elif obs_type == 'visible':
            camera_ids = [self.num_static_cameras + x for x in self.CAMERA_NUM]
            visible_objects = {}
            for cam_id in camera_ids:
                response = self.comm.get_visible_objects(cam_id)
                if response[0]:
                    visible_objects.update(response[1])
            return visible_objects

        elif obs_type == 'image':
            if 'mode' in info:
                current_mode = info['mode']
            else:
                current_mode = 'normal'

            camera_ids = [self.num_static_cameras + self.CAMERA_NUM[0], self.num_static_cameras + self.CAMERA_NUM[1]]
            if 'image_width' in info:
                image_width = info['image_width']
                image_height = info['image_height']
            else:
                image_width, image_height = self.default_image_width, self.default_image_height


            s, images = self.comm.camera_image(camera_ids, mode=current_mode, image_width=image_width, image_height=image_height)
            if not s:
                pdb.set_trace()

            if 'image_type' in info and info['image_type'] == 'concat':
                np_img = []
                for image in images:
                    np_img.append(np.array(image))
                images = cv2.vconcat(np_img)

            return images
        else:
            raise NotImplementedError
