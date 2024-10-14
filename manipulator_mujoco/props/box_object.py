from .primitive import Primitive
import numpy as np
import random


class Box(Primitive):
    """
    A class representing a box object in a simulation environment.
    """
    
    def __init__(self, size, pos, rgba, friction, n=4):
        self.possible_grasps = [
            [0.15, 0, 0, 0.500, 0.500, -0.500, -0.500],
            [0.15, 0.058, 0, 0.500, 0.500, -0.500, -0.500],
            [0.15, -0.058, 0, 0.500, 0.500, -0.500, -0.500],

            [0, -0.185, 0, 0, 0, -0.707, -0.707],
            [0, 0.185, 0, 0, 0, 0.707, -0.707],

            [0, 0.185, 0, 0.5, 0.5, 0.5, -0.5],
            [0, -0.185, 0, 0.5, -0.5, 0.5, 0.5],

            [-0.15, 0, 0, 0.500, 0.500, 0.500, 0.500],
            [-0.15, 0.058, 0, 0.500, 0.500, 0.500, 0.500],
            [-0.15, -0.058, 0, 0.500, 0.500, 0.500, 0.500],
        ]
        
        
        super(Box, self).__init__('box', size, pos, rgba, friction, n)
        

    def generate_valid_grasps(self, left_gripper_pos, right_gripper_pos, grasp_hand, object_pos, goal_grasp, currentlygrasped):
        """
        Return the top n valid grasps for the primitive object.
        """
        left_dist = np.linalg.norm(left_gripper_pos - object_pos)
        right_dist = np.linalg.norm(right_gripper_pos - object_pos)

        valid_hand = 0 if grasp_hand else 1

        # handle distance checking(basically in the case that this function returns 0 grasps, the step function should take a move action)

        # score the grasps(prune the grasps and return the best ones)
        # this should take into account the "goal" grasp



        ret_grasps = self.possible_grasps

        ret_grasps = [np.insert(grasp, 0, int(valid_hand)) for grasp in ret_grasps]


        # This removes the goal grasp and the currently grasped grasp from the list of possible grasps
        ret_grasps = [grasp for grasp in ret_grasps if (grasp != goal_grasp).any() and (grasp[1:] != currentlygrasped).any()]

        if currentlygrasped is not None:
            # this removes grasps that coincide with the currently grasped grasp(same position, different orientation)
            ret_grasps = [grasp for grasp in ret_grasps if (grasp[1:4] != currentlygrasped[:3]).any()]

        random.shuffle(ret_grasps)


        ret_grasps_prime = ret_grasps[:self.n]


        if len(ret_grasps_prime) < self.n:
            # basically need to implement a way to add dummy grasps and mask the output
            raise ValueError("Not enough grasps to return")


        # if the goal grasp is valid, replace a random grasp with the goal grasp
        if valid_hand == int(goal_grasp[0]):

            ret_grasps_prime[np.random.randint(self.n)] = goal_grasp
        

        return np.asarray(ret_grasps_prime)

    @property
    def grasps(self):
        """Returns the possible grasps for the primitive."""
        return self.possible_grasps
