<h1 align="center">Deep Q-Learning for 2048</h1>

<h2>Description</h2>

<p>
Initially, I tried implementing Q-Learning in Java using a tabular approach (https://github.com/Bloodaxe90/2048-Q-Learning), but I eventualy realized that this method only works well for small state action spaces and 2048’s is far too high for that approach.
</p>

<p>
After this failure I realised I would finally need to get into neural networks an so I learnt Python due to its larger community and mahcine learning recources. After doing this I implemented some other smaller projects to eventually build up the skills to finally re-attempt the 2048 project using Deep Q-Learning.
</p>

<p>
As I was used to JavaFX I also decided to learn and use QTDesigner for the UI of the game.
<\p>

<h2>Results</h2>

<p>
In the <em>Inference Notebook</em> section, you can see the results of the trained agent playing 2048, reaching a score of 2048. With longer training, the agent could potentially reach this score more consistently.
</p>

![image](https://github.com/user-attachments/assets/9ce39a63-9046-4e6b-b6b4-41f107b993a4)


<p>
The baseline results where the agent plays using a random policy are shown at the bottom of the <em>Experiment Notebook</em>.
</p>

This was my second attempt at implimenting Q-Learning for 2048, 
a while ago i treid to implement it withought neural networks in java as 
i had not yet realised that a tabular aproach only works for small state
action spaces. 

After this failure I started getting into neural networks, I first switched from java
to python as python has a larger comunity and resources for machine learning. Then i set about 
learning pytorch and then implemented my first deep neural net that set about to classify 6 types of images

This was a great success and I learnt alot along the way, after that I set to work on this project and finally 
successfully implemented deep q leanring on 2048

In the inference notebook section you can see the results of my trained 30000 episode agent and that it reached a score of 2048, 
if trained for longer the agent could reach this value more consitently, the base line results where the agent uses a random policy can 
be seen at the bottom of the experiment notebook


