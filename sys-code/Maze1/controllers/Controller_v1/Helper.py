import math






class HelperMethods:
    @staticmethod
    def clamp(value, low, high):
        """Clamp 'value' so it never goes below 'low' or above 'high'.

        Used everywhere a physical quantity must stay within safe limits,
        e.g. wheel speed, sensor range, pixel index.
        """
        return max(low, min(high, value))



    @staticmethod
    def wrap_angle(angle):
        """Normalise any angle (radians) into the range [-pi, +pi].

        Without this, heading errors like 'I need to turn 359 degrees right'
        would be computed instead of the correct '1 degree left', which would
        cause the robot to spin endlessly.
        """
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle
