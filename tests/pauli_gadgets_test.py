# reserved space for uni test for the circuit

# import unittest
# import sys
# import numpy as np

# # change below path to the local directory
# sys.path.append("/Users/peteryang/Downloads/MRI/Gitlab/pulse-neural/BlochSim_CK")

# # import functions/class to be tested
# from quaternion_rotation import BlochSim

# class TestRotation(unittest.TestCase):
#     ## Test 1. single output blochsim function, 90deg x rot, starting along z
#     def test_z_axis(self):
#         BlochSim_for_test = BlochSim(np.array([0,0,1]))
#         # final_Ms for result naming
#         result_rotation = BlochSim_for_test.rotation(True)
#         message = "test 1 Not small enough"
#         self.assertLess(result_rotation[2].real, 0.001, message)

#     ## Test 2. single output blochsim function, 90deg x rot, starting along x
#     def test_x_axis(self):
#         BlochSim_for_test = BlochSim(np.array([1,0,0]))
#         result_rotation = BlochSim_for_test.rotation(True)
#         message = "test 2 Not small enough"
#         self.assertLess(result_rotation[0].imag, 0.001, message)

#     ## Test 3. single output blochsim function, 90deg x rot, starting along y
#     def test_y_axis(self):
#         BlochSim_for_test = BlochSim(np.array([0,1,0]))
#         result_rotation = BlochSim_for_test.rotation(True)
#         message = "test 3 Not small enough"
#         self.assertLess(result_rotation[0].imag, 0.001, message)

#     ## Test 4. single output blochsim function, 90deg x rot, starting along y at 50% mag
#     def test_y_half_axis(self):
#         BlochSim_for_test = BlochSim(np.array([0,0.5,0]))
#         result_rotation = BlochSim_for_test.rotation(True)
#         message = "test 4 Not small enough"
#         self.assertLess(result_rotation[0].imag, 0.001, message)

#     ## Test 5. single output blochsim function, 90deg x rot
#     def test_random_axis(self):
#         BlochSim_for_test = BlochSim(np.array([0.7,0.5,0.5]))
#         result_rotation = BlochSim_for_test.rotation(True)
#         message = "test 5 Not small enough"
#         self.assertEqual(round(result_rotation[0].real, 2), 0.7, message)
#         self.assertEqual(round(result_rotation[1].imag, 2), 0.5, message)
#         self.assertEqual(round(result_rotation[2].real, 2), 0.5, message)


# if __name__ == '__main__':
#     unittest.main()