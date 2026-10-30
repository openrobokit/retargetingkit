"""Noitom MocapApi ctypes bindings.

Vendored from mocap-manager/third_party/mocap_ros_py/mocap_robotapi.py
and imported into this repository by commit
3ed57a64465dff24955f1c39d5be1b68f5aad6d6. The API interface versions below
define the native ABI. Keep this file in sync with the Noitom runtime library
instead of refactoring it as application code.
"""

from collections import namedtuple
from ctypes import (
    CFUNCTYPE,
    POINTER,
    Structure,
    Union,
    c_bool,
    c_char_p,
    c_double,
    c_float,
    c_int32,
    c_uint16,
    c_uint32,
    c_uint64,
    cdll,
    pointer,
    sizeof,
)

from axis_studio_sdk_runtime import library_path


MocapApi = cdll.LoadLibrary(str(library_path()))

MCPError = namedtuple(
    "EMCPError",
    [
        "NoError",
        "MoreEvent",
        "InsufficientBuffer",
        "InvalidObject",
        "InvalidHandle",
        "InvalidParameter",
        "NotSupported",
        "IgnoreUDPSettings",
        "IgnoreTCPSettings",
        "IgnoreBvhSettings",
        "JointNotFound",
        "WithoutTransformation",
        "NoneMessage",
        "NoneParent",
        "NoneChild",
        "AddressInUse",
    ],
)._make(range(16))

MCPJointTag = namedtuple(
    "EMCPJointTag",
    [
        "Hips",
        "RightUpLeg",
        "RightLeg",
        "RightFoot",
        "LeftUpLeg",
        "LeftLeg",
        "LeftFoot",
        "Spine",
        "Spine1",
        "Spine2",
        "Neck",
        "Neck1",
        "Head",
        "RightShoulder",
        "RightArm",
        "RightForeArm",
        "RightHand",
        "RightHandThumb1",
        "RightHandThumb2",
        "RightHandThumb3",
        "RightInHandIndex",
        "RightHandIndex1",
        "RightHandIndex2",
        "RightHandIndex3",
        "RightInHandMiddle",
        "RightHandMiddle1",
        "RightHandMiddle2",
        "RightHandMiddle3",
        "RightInHandRing",
        "RightHandRing1",
        "RightHandRing2",
        "RightHandRing3",
        "RightInHandPinky",
        "RightHandPinky1",
        "RightHandPinky2",
        "RightHandPinky3",
        "LeftShoulder",
        "LeftArm",
        "LeftForeArm",
        "LeftHand",
        "LeftHandThumb1",
        "LeftHandThumb2",
        "LeftHandThumb3",
        "LeftInHandIndex",
        "LeftHandIndex1",
        "LeftHandIndex2",
        "LeftHandIndex3",
        "LeftInHandMiddle",
        "LeftHandMiddle1",
        "LeftHandMiddle2",
        "LeftHandMiddle3",
        "LeftInHandRing",
        "LeftHandRing1",
        "LeftHandRing2",
        "LeftHandRing3",
        "LeftInHandPinky",
        "LeftHandPinky1",
        "LeftHandPinky2",
        "LeftHandPinky3",
        "Spine3",
        "JointsCount",
    ],
)._make([x for x in range(61)])

MCPRigidBodyHandle = c_uint64


class MCPRigidBody(object):
    IMCPRigidBodyApi_Version = c_char_p(b"PROC_TABLE:IMCPRigidBody_001")

    class MCPRigidBodyApi(Structure):
        _fields_ = [
            (
                "GetRigidBodyRotation",
                CFUNCTYPE(
                    c_int32, POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), MCPRigidBodyHandle
                ),
            ),
            (
                "GetRigidBodyPosition",
                CFUNCTYPE(c_int32, POINTER(c_float), POINTER(c_float), POINTER(c_float), MCPRigidBodyHandle),
            ),
            ("GetRigidBodyStatus", CFUNCTYPE(c_int32, POINTER(c_int32), MCPRigidBodyHandle)),
            ("GetRigidBodyId", CFUNCTYPE(c_int32, POINTER(c_int32), MCPRigidBodyHandle)),
        ]

    api = POINTER(MCPRigidBodyApi)()

    def __init__(self, rigid_body_handle):
        if not self.api:
            err = MocapApi.MCPGetGenericInterface(self.IMCPRigidBodyApi_Version, pointer(self.api))
            if err != MCPError.NoError:
                raise RuntimeError("Can not get MCPRigidBody interface: {0}".format(MCPError._fields[err]))
        self.handle = rigid_body_handle

    def get_rotation(self):
        x, y, z, w = c_float(), c_float(), c_float(), c_float()
        err = self.api.contents.GetRigidBodyRotation(pointer(x), pointer(y), pointer(z), pointer(w), self.handle)
        if err != MCPError.NoError:
            raise RuntimeError("Can not get rigid body rotation: {0}".format(MCPError._fields[err]))
        return w.value, x.value, y.value, z.value

    def get_position(self):
        x, y, z = c_float(), c_float(), c_float()
        err = self.api.contents.GetRigidBodyPosition(pointer(x), pointer(y), pointer(z), self.handle)
        if err != MCPError.NoError:
            raise RuntimeError("Can not get rigid body position: {0}".format(MCPError._fields[err]))
        return x.value, y.value, z.value


MCPSensorModuleHandle = c_uint64


class MCPSensorModule(object):
    IMCPSensorModuleApi_Version = c_char_p(b"PROC_TABLE:IMCPSensorModule_001")

    class MCPSensorModuleApi(Structure):
        _fields_ = [
            (
                "GetSensorModulePosture",
                CFUNCTYPE(
                    c_int32,
                    POINTER(c_float),
                    POINTER(c_float),
                    POINTER(c_float),
                    POINTER(c_float),
                    MCPSensorModuleHandle,
                ),
            ),
            (
                "GetSensorModuleAngularVelocity",
                CFUNCTYPE(c_int32, POINTER(c_float), POINTER(c_float), POINTER(c_float), MCPSensorModuleHandle),
            ),
            (
                "GetSensorModuleAcceleratedVelocity",
                CFUNCTYPE(c_int32, POINTER(c_float), POINTER(c_float), POINTER(c_float), MCPSensorModuleHandle),
            ),
        ]

    api = POINTER(MCPSensorModuleApi)()

    def __init__(self, sensor_handle):
        if not self.api:
            err = MocapApi.MCPGetGenericInterface(self.IMCPSensorModuleApi_Version, pointer(self.api))
            if err != MCPError.NoError:
                raise RuntimeError("Can not get MCPSensorModule interface: {0}".format(MCPError._fields[err]))
        self.handle = sensor_handle

    def get_posture(self):
        x, y, z, w = c_float(), c_float(), c_float(), c_float()
        err = self.api.contents.GetSensorModulePosture(pointer(x), pointer(y), pointer(z), pointer(w), self.handle)
        if err != MCPError.NoError:
            raise RuntimeError("Can not get sensor module posture: {0}".format(MCPError._fields[err]))
        return w.value, x.value, y.value, z.value

    def get_angular_velocity(self):
        x, y, z = c_float(), c_float(), c_float()
        err = self.api.contents.GetSensorModuleAngularVelocity(pointer(x), pointer(y), pointer(z), self.handle)
        if err != MCPError.NoError:
            raise RuntimeError("Can not get sensor module angular velocity: {0}".format(MCPError._fields[err]))
        return x.value, y.value, z.value

    def get_accelerated_velocity(self):
        x, y, z = c_float(), c_float(), c_float()
        err = self.api.contents.GetSensorModuleAcceleratedVelocity(pointer(x), pointer(y), pointer(z), self.handle)
        if err != MCPError.NoError:
            raise RuntimeError("Can not get sensor module accelerated velocity: {0}".format(MCPError._fields[err]))
        return x.value, y.value, z.value


MCPBodyPartHandle = c_uint64


class MCPBodyPart(object):
    IMCPBodyPartApi_Version = c_char_p(b"PROC_TABLE:IMCPBodyPart_001")

    class MCPBodyPartApi(Structure):
        _fields_ = [
            (
                "GetJointPosition",
                CFUNCTYPE(c_int32, POINTER(c_float), POINTER(c_float), POINTER(c_float), MCPBodyPartHandle),
            ),
            (
                "GetJointDisplacementSpeed",
                CFUNCTYPE(c_int32, POINTER(c_float), POINTER(c_float), POINTER(c_float), MCPBodyPartHandle),
            ),
            (
                "GetBodyPartPosture",
                CFUNCTYPE(
                    c_int32, POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), MCPBodyPartHandle
                ),
            ),
        ]

    api = POINTER(MCPBodyPartApi)()

    def __init__(self, body_part_handle):
        if not self.api:
            err = MocapApi.MCPGetGenericInterface(self.IMCPBodyPartApi_Version, pointer(self.api))
            if err != MCPError.NoError:
                raise RuntimeError("Can not get MCPBodyPartApi interface: {0}".format(MCPError._fields[err]))
        self.handle = body_part_handle

    def get_position(self):
        x, y, z = c_float(), c_float(), c_float()
        err = self.api.contents.GetJointPosition(pointer(x), pointer(y), pointer(z), self.handle)
        if err != MCPError.NoError:
            raise RuntimeError("Can not get joint position: {0}".format(MCPError._fields[err]))
        return x.value, y.value, z.value

    def get_displacement_speed(self):
        x, y, z = c_float(), c_float(), c_float()
        err = self.api.contents.GetJointDisplacementSpeed(pointer(x), pointer(y), pointer(z), self.handle)
        if err != MCPError.NoError:
            raise RuntimeError("Can not get joint displacement speed: {0}".format(MCPError._fields[err]))
        return x.value, y.value, z.value

    def get_posture(self):
        x, y, z, w = c_float(), c_float(), c_float(), c_float()
        err = self.api.contents.GetBodyPartPosture(pointer(x), pointer(y), pointer(z), pointer(w), self.handle)
        if err != MCPError.NoError:
            raise RuntimeError("Can not get body part posture: {0}".format(MCPError._fields[err]))
        return w.value, x.value, y.value, z.value


MCPJointHandle = c_uint64


class MCPJoint(object):
    IMCPJointApi_Version = c_char_p(b"PROC_TABLE:IMCPJoint_003")

    class MCPJointApi(Structure):
        _fields_ = [
            ("GetJointName", CFUNCTYPE(c_int32, POINTER(c_char_p), MCPJointHandle)),
            (
                "GetJointLocalRotation",
                CFUNCTYPE(
                    c_int32, POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), MCPJointHandle
                ),
            ),
            (
                "GetJointLocalRotationByEuler",
                CFUNCTYPE(c_int32, POINTER(c_float), POINTER(c_float), POINTER(c_float), MCPJointHandle),
            ),
            (
                "GetJointLocalTransformation",
                CFUNCTYPE(c_int32, POINTER(c_float), POINTER(c_float), POINTER(c_float), MCPJointHandle),
            ),
            (
                "GetJointDefaultLocalTransformation",
                CFUNCTYPE(c_int32, POINTER(c_float), POINTER(c_float), POINTER(c_float), MCPJointHandle),
            ),
            ("GetJointChild", CFUNCTYPE(c_int32, POINTER(MCPJointHandle), POINTER(c_uint32), MCPJointHandle)),
            ("GetJointBodyPart", CFUNCTYPE(c_int32, POINTER(MCPBodyPartHandle), MCPJointHandle)),
            ("GetJointSensorModule", CFUNCTYPE(c_int32, POINTER(MCPSensorModuleHandle), MCPJointHandle)),
            ("GetJointTag", CFUNCTYPE(c_int32, POINTER(c_int32), MCPJointHandle)),
            ("GetJointNameByTag", CFUNCTYPE(c_int32, POINTER(c_char_p), c_int32)),
            ("GetJointChildJointTag", CFUNCTYPE(c_int32, POINTER(c_int32), POINTER(c_uint32), c_int32)),
            ("GetJointParentJointTag", CFUNCTYPE(c_int32, POINTER(c_int32), c_int32)),
        ]

    api = POINTER(MCPJointApi)()

    def __init__(self, joint_handle):
        if not self.api:
            err = MocapApi.MCPGetGenericInterface(self.IMCPJointApi_Version, pointer(self.api))
            if err != MCPError.NoError:
                raise RuntimeError("Can not get MCPJointApi interface: {0}".format(MCPError._fields[err]))
        self.handle = joint_handle

    def get_name(self):
        joint_name = c_char_p()
        err = self.api.contents.GetJointName(pointer(joint_name), self.handle)
        if err != MCPError.NoError:
            raise RuntimeError("Can not get joint name: {0}".format(MCPError._fields[err]))
        return str(joint_name.value or b"", encoding="utf8")

    def get_local_rotation(self):
        x, y, z, w = c_float(), c_float(), c_float(), c_float()
        err = self.api.contents.GetJointLocalRotation(pointer(x), pointer(y), pointer(z), pointer(w), self.handle)
        if err != MCPError.NoError:
            raise RuntimeError("Can not get joint local rotation: {0}".format(MCPError._fields[err]))
        return w.value, x.value, y.value, z.value

    def get_local_position(self):
        x, y, z = c_float(), c_float(), c_float()
        err = self.api.contents.GetJointLocalTransformation(pointer(x), pointer(y), pointer(z), self.handle)
        if err != MCPError.NoError:
            return None
        return x.value, y.value, z.value

    def get_children(self):
        joint_count = c_uint32()
        err = self.api.contents.GetJointChild(POINTER(MCPJointHandle)(), pointer(joint_count), self.handle)
        if err != MCPError.NoError:
            raise RuntimeError("Can not get joint child count: {0}".format(MCPError._fields[err]))
        joint_handles = (MCPJointHandle * joint_count.value)()
        err = self.api.contents.GetJointChild(joint_handles, pointer(joint_count), self.handle)
        if err != MCPError.NoError:
            raise RuntimeError("Can not get joint child: {0}".format(MCPError._fields[err]))
        return [MCPJoint(joint_handles[i]) for i in range(joint_count.value)]

    def get_body_part(self):
        body_part_handle = MCPBodyPartHandle()
        err = self.api.contents.GetJointBodyPart(pointer(body_part_handle), self.handle)
        if err != MCPError.NoError:
            raise RuntimeError("Can not get joint body part: {0}".format(MCPError._fields[err]))
        return MCPBodyPart(body_part_handle)

    def get_sensor_module(self):
        sensor_handle = MCPSensorModuleHandle()
        err = self.api.contents.GetJointSensorModule(pointer(sensor_handle), self.handle)
        if err != MCPError.NoError:
            raise RuntimeError("Can not get sensor module: {0}".format(err))
        return MCPSensorModule(sensor_handle)

    def get_tag(self):
        tag = c_int32()
        err = self.api.contents.GetJointTag(pointer(tag), self.handle)
        if err != MCPError.NoError:
            raise RuntimeError("Can not get joint tag: {0}".format(MCPError._fields[err]))
        return tag.value


MCPAvatarHandle = c_uint64


class MCPAvatar(object):
    IMCPAvatarApi_Version = c_char_p(b"PROC_TABLE:IMCPAvatar_003")

    class MCPAvatarApi(Structure):
        _fields_ = [
            ("GetAvatarIndex", CFUNCTYPE(c_int32, POINTER(c_uint32), MCPAvatarHandle)),
            ("GetAvatarRootJoint", CFUNCTYPE(c_int32, POINTER(MCPJointHandle), MCPAvatarHandle)),
            ("GetAvatarJoints", CFUNCTYPE(c_int32, POINTER(MCPJointHandle), POINTER(c_uint32), MCPAvatarHandle)),
            ("GetAvatarJointByName", CFUNCTYPE(c_int32, c_char_p, POINTER(MCPJointHandle), MCPAvatarHandle)),
            ("GetAvatarName", CFUNCTYPE(c_int32, POINTER(c_char_p), MCPAvatarHandle)),
            (
                "GetAvatarRigidBodies",
                CFUNCTYPE(c_int32, POINTER(MCPRigidBodyHandle), POINTER(c_uint32), MCPAvatarHandle),
            ),
            ("GetAvatarJointHierarchy", CFUNCTYPE(c_int32, POINTER(c_char_p))),
            ("GetAvatarPostureIndex", CFUNCTYPE(c_int32, POINTER(c_uint32), POINTER(MCPAvatarHandle))),
            (
                "GetAvatarPostureTimeCode",
                CFUNCTYPE(
                    c_int32,
                    POINTER(c_uint32),
                    POINTER(c_uint32),
                    POINTER(c_uint32),
                    POINTER(c_uint32),
                    POINTER(MCPAvatarHandle),
                ),
            ),
        ]

    api = POINTER(MCPAvatarApi)()

    def __init__(self, avatar_handle):
        if not self.api:
            err = MocapApi.MCPGetGenericInterface(self.IMCPAvatarApi_Version, pointer(self.api))
            if err != MCPError.NoError:
                raise RuntimeError("Can not get MCPAvatar interface: {0}".format(MCPError._fields[err]))
        self.handle = avatar_handle

    def get_root_joint(self):
        joint_handle = MCPJointHandle()
        err = self.api.contents.GetAvatarRootJoint(pointer(joint_handle), self.handle)
        if err != MCPError.NoError:
            raise RuntimeError("Can not get avatar root joint: {0}".format(err))
        return MCPJoint(joint_handle)

    def get_name(self):
        avatar_name = c_char_p()
        err = self.api.contents.GetAvatarName(pointer(avatar_name), self.handle)
        if err != MCPError.NoError:
            raise RuntimeError("Can not get avatar name: {0}".format(MCPError._fields[err]))
        return str(avatar_name.value or b"", encoding="utf8")


class MCPEventDataReserved(Structure):
    _fields_ = [
        ("reserved0", c_uint64),
        ("reserved1", c_uint64),
        ("reserved2", c_uint64),
        ("reserved3", c_uint64),
        ("reserved4", c_uint64),
        ("reserved5", c_uint64),
    ]


class MCPEventData(Union):
    _fields_ = [("reserved", MCPEventDataReserved), ("avatar_handle", MCPAvatarHandle), ("error", c_int32)]


MCPEventType = namedtuple("EMCPEventType", ["InvalidEvent", "AvatarUpdated", "RigidBodyUpdated", "Error"])(
    0, 256, 512, 768
)


class MCPEvent(Structure):
    _fields_ = [("size", c_uint32), ("event_type", c_int32), ("timestamp", c_double), ("event_data", MCPEventData)]


MCPSettingsHandle = c_uint64


class MCPSettings(object):
    IMCPSettingsApi_Version = c_char_p(b"PROC_TABLE:IMCPSettings_001")

    class MCPSettingsApi(Structure):
        _fields_ = [
            ("CreateSettings", CFUNCTYPE(c_int32, POINTER(MCPSettingsHandle))),
            ("DestroySettings", CFUNCTYPE(c_int32, MCPSettingsHandle)),
            ("SetSettingsUDP", CFUNCTYPE(c_int32, c_uint16, MCPSettingsHandle)),
            ("SetSettingsTCP", CFUNCTYPE(c_int32, c_char_p, c_uint16, MCPSettingsHandle)),
            ("SetSettingsBvhRotation", CFUNCTYPE(c_int32, c_int32, MCPSettingsHandle)),
            ("SetSettingsBvhDisplacement", CFUNCTYPE(c_int32, c_int32, MCPSettingsHandle)),
            ("SetSettingsBvhData", CFUNCTYPE(c_int32, c_int32, MCPSettingsHandle)),
            ("SetSettingsCalcData", CFUNCTYPE(c_int32, MCPSettingsHandle)),
        ]

    api = POINTER(MCPSettingsApi)()

    def __init__(self):
        if not self.api:
            err = MocapApi.MCPGetGenericInterface(self.IMCPSettingsApi_Version, pointer(self.api))
            if err != MCPError.NoError:
                raise RuntimeError("Can not get MCPSettings interface: {0}".format(MCPError._fields[err]))
        self.handle = MCPSettingsHandle()
        err = self.api.contents.CreateSettings(pointer(self.handle))
        if err != MCPError.NoError:
            raise RuntimeError("Can not create settings: {0}".format(MCPError._fields[err]))

    def __del__(self):
        self.api.contents.DestroySettings(self.handle)

    def set_udp(self, local_port):
        err = self.api.contents.SetSettingsUDP(c_uint16(local_port), self.handle)
        if err != MCPError.NoError:
            raise RuntimeError("Can not set udp port of {0}: {1}".format(local_port, MCPError._fields[err]))

    def set_calc_data(self):
        err = self.api.contents.SetSettingsCalcData(self.handle)
        if err != MCPError.NoError:
            raise RuntimeError("Can not set calc data: {0}".format(MCPError._fields[err]))


MCPApplicationHandle = c_uint64


class MCPApplication(object):
    IMCPApplicationApi_Version = c_char_p(b"PROC_TABLE:IMCPApplication_002")

    class MCPApplicationApi(Structure):
        _fields_ = [
            ("CreateApplication", CFUNCTYPE(c_int32, POINTER(MCPApplicationHandle))),
            ("DestroyApplication", CFUNCTYPE(c_int32, MCPApplicationHandle)),
            ("SetApplicationSettings", CFUNCTYPE(c_int32, MCPSettingsHandle, MCPApplicationHandle)),
            ("SetApplicationRenderSettings", CFUNCTYPE(c_int32, c_uint64, MCPApplicationHandle)),
            ("OpenApplication", CFUNCTYPE(c_int32, MCPApplicationHandle)),
            ("EnableApplicationCacheEvents", CFUNCTYPE(c_int32, MCPApplicationHandle)),
            ("DisableApplicationCacheEvents", CFUNCTYPE(c_int32, MCPApplicationHandle)),
            ("ApplicationCacheEventsIsEnabled", CFUNCTYPE(c_int32, POINTER(c_bool), MCPApplicationHandle)),
            ("CloseApplication", CFUNCTYPE(c_int32, MCPApplicationHandle)),
            (
                "GetApplicationRigidBodies",
                CFUNCTYPE(c_int32, POINTER(c_uint64), POINTER(c_uint32), MCPApplicationHandle),
            ),
            ("GetApplicationAvatars", CFUNCTYPE(c_int32, POINTER(c_uint64), POINTER(c_uint32), MCPApplicationHandle)),
            (
                "PollApplicationNextEvent",
                CFUNCTYPE(c_int32, POINTER(MCPEvent), POINTER(c_uint32), MCPApplicationHandle),
            ),
        ]

    api = POINTER(MCPApplicationApi)()

    def __init__(self):
        if not self.api:
            err = MocapApi.MCPGetGenericInterface(self.IMCPApplicationApi_Version, pointer(self.api))
            if err != MCPError.NoError:
                raise RuntimeError("Can not get IMCPApplication interface: {0}".format(MCPError._fields[err]))
        self._handle = MCPApplicationHandle()
        err = self.api.contents.CreateApplication(pointer(self._handle))
        if err != MCPError.NoError:
            raise RuntimeError("Can not create application: {0}".format(MCPError._fields[err]))

    def __del__(self):
        self.api.contents.DestroyApplication(self._handle)

    def set_settings(self, settings):
        err = self.api.contents.SetApplicationSettings(settings.handle, self._handle)
        if err != MCPError.NoError:
            raise RuntimeError("Can not set application settings: {0}".format(MCPError._fields[err]))

    def open(self):
        err = self.api.contents.OpenApplication(self._handle)
        return err == MCPError.NoError, MCPError._fields[err]

    def close(self):
        err = self.api.contents.CloseApplication(self._handle)
        return err == MCPError.NoError, MCPError._fields[err]

    def poll_next_event(self):
        evt_count = c_uint32(0)
        err = self.api.contents.PollApplicationNextEvent(POINTER(MCPEvent)(), pointer(evt_count), self._handle)
        if err != MCPError.NoError:
            raise RuntimeError("Can not poll application next event: {0}".format(MCPError._fields[err]))
        if evt_count.value == 0:
            return []
        evt_array = (MCPEvent * evt_count.value)()
        for i in range(evt_count.value):
            evt_array[i].size = sizeof(MCPEvent)
        err = self.api.contents.PollApplicationNextEvent(evt_array, pointer(evt_count), self._handle)
        if err != MCPError.NoError or evt_count.value == 0:
            return []
        return [evt_array[i] for i in range(evt_count.value)]
