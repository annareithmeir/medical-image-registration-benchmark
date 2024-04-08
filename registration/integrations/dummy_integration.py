from core.registration_interface import RegistrationInterface


class DummyIntegration(RegistrationInterface):
    def __init__(self, parameter):
        # Initialization code
        pass

    def register(self, fixed_image, moving_image):
        # Implementation
        pass

    def get_transformation(self):
        # Return transformation model
        pass
