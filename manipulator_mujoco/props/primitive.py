from dm_control import mjcf
import numpy as np
import random

class Primitive(object):
    """
    A base class representing a primitive object in a simulation environment.
    """

    def __init__(self, type, size, pos, rgba, friction, n=4):
        """
        Initialize the Primitive object.

        Args:
            type: The type of the geometric shape (e.g., 'box', 'sphere').
            size: The size of the shape.
            pos: The initial position of the shape.
            rgba: The color and transparency of the shape.
            friction: The friction properties of the shape.
        """
        self._mjcf_model = mjcf.RootElement()
        
        # Add a body to the worldbody
        self._body = self._mjcf_model.worldbody.add("body")

        # Add a geometric element to the body
        self._geom = self._body.add("geom", type=type, size=size, pos=pos, rgba=rgba, friction=friction)

        # Set the possible grasps for the primitive
        self.n = n

    @property
    def geom(self):
        """Returns the primitive's geom, e.g., to change color or friction."""
        return self._geom
    
    @property
    def body(self):
        """Returns the primitive's body."""
        return self._body

    @property
    def mjcf_model(self):
        """Returns the primitive's MJCF model."""
        return self._mjcf_model

    
    def generate_valid_grasps(self, left_gripper_pos, right_gripper_pos, grasp_hand, object_pos, goal_grasp, currentlygrasped):
        """
        Return the top n valid grasps for the primitive object.
        """
        raise NotImplementedError
