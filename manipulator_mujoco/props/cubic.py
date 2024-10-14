

class Cubic():
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
        
    
    def generate_valid_grasps(self, left_gripper_pos, right_gripper_pos, grasp_hand, object_pos, goal_grasp, currentlygrasped):
        pass

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

    @property
    def grasps(self):
        """Returns the possible grasps for the primitive."""
        return self.possible_grasps
