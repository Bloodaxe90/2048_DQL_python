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
<h3>Baseline</h3>

![image](https://github.com/user-attachments/assets/849c20c3-d3c3-4754-8a0f-6974410169d9)
<p>
These baseline results are when the agent played using a random policy. This image can originally be found in the Experiment Notebook.
</p>
<h3>Final Results</h3>

![image](https://github.com/user-attachments/assets/9ce39a63-9046-4e6b-b6b4-41f107b993a4)

<p>
After a lot of testing I trained my model for 30,000 episodes which took about 4 days.
These results show an amazing improvement from the baseline with the agent even reaching a score of 2048 occasionally, I am sure that with more training an agent would be able to consistently reach a value of 2048. The original image of the results can be found in the Inference Notebook.
</p>



