# Calibrating_IMU
Calibrating IMU with linear regression and analytical solution to non linear system of equations

## Description

This is a unique method of calibrating an IMU. The IMU must be rotated parallel to the ground to establish the plane of movement along the x-y plane.

That data is first fit to a plane, then fit to a circle with a linear then non linear regression.

The circle parameters are then used to find the yaw of incoming datapoints
