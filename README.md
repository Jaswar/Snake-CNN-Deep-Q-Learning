# Snake-CNN-Deep-Q-Learning

  This is a repository containing a game of popular Snake that I have built on my own and an AI to play it. AI itself uses Convolutional Neural Network and a method called Deep-Q-Learning to get better at this game. Game is based on a grid of a certain size with a Snake coloured by default in white, apples that it is supposed to eat which are coloured in red and the background is black:

Required libraries:
- tensorflow (recommended tensorflow-gpu and theano for faster computing)
- keras
- numpy
- pygame (version 1.9.4 not sure if higher versions work)
- numpy
- matplotlib

My specs:
-i7-7700HQ Processor
-16GB RAM
-NVIDIA Quadro M1200 4GB graphics card

Training description:
  Training starts with epsilon set to initial value which then decreases every epoch by some amount until it reaches minimum value. The size of the grid can be changed, but remember that bigger grid will mean slower or no training. The default is set to 10x10. Neural network is a convolutional network which means that it searches for some features called filters. Input is array with the same size as the grid (this array describes image and every cell in game), which is image of game at certain point in time. This means that AI learns from image. Since however, AI cannot guess which way it is going, it takes on input certain number of previous images (default is 4). When it comes to the rewards: -0.1 (might be too little) is default reward, -1 is for hitting itself or the wall and 2 for eating an apple. AI can take 4 actions corresponding to every direction it can move in. 
  
Results:
  You can see training results on a 10x10 grid in 'stats.png'. After some more training it reached an average result of 12. Training it to this moment took me around 17 hours.
