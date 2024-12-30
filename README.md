![image](https://github.com/user-attachments/assets/e0f61c36-4088-424b-b92b-0c3d25b674e5)This was my second attempt at implimenting Q-Learning for 2048, 
a while ago i treid to implement it withought neural networks in java as 
i had not yet realised that a tabular aproach only works for small state
action spaces. 

After this failure I started getting into neural networks, I first switched from java
to python as python has a larger comunity and resources for machine learning. Then i set about 
learning pytorch and then implemented my first deep neural net that set about to classify 6 types of images

This was a great success and I learnt alot along the way, after that I set to work on this project and finally 
successfully implemented deep q leanring on 2048

Below is the baseline results using a random policy for the agent.
![image](https://github.com/user-attachments/assets/edb0bdb7-c0f8-4eb8-8fc7-7c30c63d9a09)

Next I testest that the network acutally learnt by training it for 200 episodes and then 7500 episodes
![image](https://github.com/user-attachments/assets/b1b7648f-ebb4-4062-8a8f-68f7ac014746)
^ 200 episodes worked well but I needed more evidence that it could learn successfully so i ran for 7500 episodes

![image](https://github.com/user-attachments/assets/85cc3ca7-fb64-438a-b3e0-28de44f793f8)
^ as you can see when trained for 7500 episodes the agent is able to reach a value of 1024

After these successses I trained for 30000 episodes in the hopes that the agent would reach a value of 2048
![image](https://github.com/user-attachments/assets/0c4898c4-9bb8-400a-9974-5e5a1a170bc0)
as seen in the image the agent could reach a vlue of 2048, showing that, if trained for longer a value of 2048 could be reached consistently

