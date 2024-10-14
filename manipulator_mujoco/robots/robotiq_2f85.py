import os
from manipulator_mujoco.robots.gripper import Gripper

_2F85_XML_left = os.path.join(
    os.path.dirname(__file__),
    '../assets/robots/robotiq_2f85/2f85_left.xml',
)

_2F85_XML_right = os.path.join(
    os.path.dirname(__file__),
    '../assets/robots/robotiq_2f85/2f85_right.xml',
)

_JOINT = 'right_driver_joint'

_ACTUATOR = 'fingers_actuator'

class Robotiq_2F85(Gripper):
    # hand = 0 for left, 1 for right
    def __init__(self, hand=0, name: str = None):

        if hand == 0:
            super().__init__(_2F85_XML_left, _JOINT, _ACTUATOR, name)
        else:
            super().__init__(_2F85_XML_right, _JOINT, _ACTUATOR, name)
        self._object_site = self._mjcf_root.find('site', 'pinch')
        self.gripper_state = False
        # self._actuator.ctrl = 1.0

    @property
    def object_site(self):
        return self._object_site
    
    def is_closed(self, physics):
        return self.gripper_state
    
    def close(self, physics):
        self.gripper_state = True
        # self._actuator.ctrl = 0.0

    def open(self, physics):
        self.gripper_state = False
        # self._actuator.ctrl = 1.0
    
    def attach_object(self, child, pos: list = [0, 0, 0], quat: list = [1, 0, 0, 0]):
        # try:
        #     child.detach()
        # except:
        #     pass
        frame = self._object_site.attach(child)
        frame.pos = pos
        frame.quat = quat
        frame.add("joint", type="hinge", damping=3.5, axis=[0,0,1])
        return frame
    
    def detach_object(self, child):
        child.detach()

            
