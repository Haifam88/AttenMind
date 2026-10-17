# AttenMind
AttenMind measures student attention from a single image. It uses three deep learning models stored together in one file (attenmind_combined.h5):    1. Yawn Detection  - MobileNetV2 - detects whether the student is yawning   2. Head Pose       - ResNet-18   - checks whether the head faces the board   3. Eye Attention   - MobileNetV2 - checks whether the eyes are open
