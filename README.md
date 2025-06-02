<h1 align="center">Deep Q-Learning for 2048</h1>

<h2>Description</h2>

<p>
This is my second attempt at implementing Q-Learning on 2048, following a previous failed project 
(https://github.com/Bloodaxe90/2048-Q-Learning)
where I used a tabular approach (more details in that repository). Thankfully, this attempt was much more successful!
</p>

<p>
This project's UI was built using PySide6 with Qt Designer, and it uses TensorBoard for inference.
</p>

<h2>Usage:</h2>
<ol>
  <li>Activate a virtual environment.</li>
  <li>Run <code>pip install -r requirements.txt</code> to install the dependencies.</li>
  <li><strong>Either:</strong></li>
  <ul>
    <li>Run <code>main.py</code> to train a model</li>
    <li>Run <code>application.py</code> to test a model play 2048 visually or play 2048 yourself.</li>
  </ul>
</ol>

<h2>Hyperparameters:</h2>

<p><strong>Hyperparameters found in <code>main.py</code>:</strong></p>
<ul>
  <li><code>EPISODES</code> (int): The number of episodes to train across.</li>
  <li>
    <code>HIDDEN_NEURONS</code> (tuple[int]): Defines the number of hidden neurons in each hidden layer. 
    The number of hidden layers is <code>len(HIDDEN_NEURONS) - 1</code>. 
    For example, <code>(128, 64, 32)</code> results in two hidden layers: the first with 128 input and 64 output neurons, 
    and the second with 64 input and 32 output neurons.
  </li>
  <li><code>REPLAY_CAPACITY</code> (int): The capacity of the replay buffer.</li>
  <li><code>BATCH_SIZE</code> (int): The number of experiences used in each training step.</li>
  <li><code>ALPHA</code> (float): The learning rate.</li>
  <li><code>GAMMA</code> (float): The discount factor.</li>
  <li><code>TRIAL_NAME</code> (str): The name of the current experiment, used as part of the filename for the TensorBoard logs.</li>
  <li>
    <code>MAIN_UPDATE_COUNT</code> (int): The number of training steps performed on the main network when an update condition is met.
  </li>
  <li><code>MAIN_UPDATE_FREQ</code> (int): The frequency (in episodes) at which the main network is updated.</li>
  <li><code>TARGET_UPDATE_FREQ</code> (int): The frequency (in episodes) at which the target network is updated from the main network.</li>
  <li>
    <code>MODEL_SAVE_NAME</code> (str): The name to save the trained model under. Leave as an empty string if the model should not be saved.
  </li>
</ul>

<p><strong>Hyperparameters found in <code>application.py</code>:</strong></p>
<ul>
  <li><code>MODEL_LOAD_NAME</code> (str): The name of the model to load and use for playing 2048.</li>
  <li>
    <code>MODEL_LOAD_HIDDEN_NEURONS</code> (tuple[int]): The hidden layer structure of the model being loaded. 
    Follows the same format as <code>HIDDEN_NEURONS</code> described above.
  </li>
</ul>

<h2>Controls:</h2>
<ul>
  <li><strong>Default Radio Button:</strong> (Disabled while the agent is autoplaying)
    <br>Allows you to play 2048 manually.</br>
    <ul>
      <li><strong>Arrow Keys:</strong> Move the number tiles in the corresponding direction.</li>
    </ul>
  </li>
  <li><strong>Q-AI Radio Button:</strong>
    <br>Enables the agent to play automatically.</br>
    <ul>
      <li><strong>S Key:</strong> Starts or stops the agent autoplaying 2048.</li>
    </ul>
  </li>
  <li><strong>Space Bar:</strong> Resets the game (Disabled while the agent is autoplaying).</li>
</ul>


  
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

<p>
Screenshot of the final UI:
</p>

![image](https://github.com/user-attachments/assets/8cf9e56e-9c4b-4616-9c26-cb9ee1429ef6)








