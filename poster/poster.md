
<style>

    @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;500;600;700&display=swap');

    * {
        font-family: 'Open Sans', sans-serif;
    }
    .header {
        background-color: #F8F8FF;
        border: 2px solid #a9b4c2;
        border-bottom: none; /* Remove the bottom border */
        border-radius: 10px 10px 0 0; /* Top left and right corners are rounded */
        padding: 20px;
    }

    .header h1 {
        color: #2b3469;
        margin: 0; /* This removes the default margin */
        font-weight: 700;
        font-size: 30px;
    }
    
  .container {
    border: 2px solid #a9b4c2;
    background-color: #F8F8FF;
    padding: 20px;
    border-top: none;
    border-radius: 0 0 10px 10px;
    margin-bottom: 10px;
    position: relative;
    font-size: 17px;
    font-weight: 500;
    color: #2C4A67;
    margin-top: -2px;
}

.container::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    box-shadow: 0 10px 10px rgba(0, 0, 0, 0.15);
    border-radius: 0 0 10px 10px;
    z-index: -1;
}

  .container h2 {
    font-weight: 500;
  }
   
  .section {
    display: flex;
    align-items: center;
  }

  .section-content {
    flex: 1;
  }

  .section-image {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .section-image img {
    max-width: 100%;
    height: auto;
  }

  .styled-table {
    margin: 0 auto; /* Center table on the page */
    border-collapse: collapse;
    font-size: 0.9em;
    box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
    width: 100%;
    border-radius: 0.8em;
    overflow: hidden;
  }

  .styled-table thead tr {
    background-color: #5763a6;
    color: #ffffff;
    text-align: center;
  }

  .styled-table th, .styled-table td {
    text-align: center;
    padding: 2px;
  }

  .styled-table tbody tr {
    border-bottom: 1px solid #dddddd;
  }

  .styled-table tbody tr:nth-of-type(even) {
    background-color: #f3f3f3;
  }

  .table-wrapper {
      margin: 0 10px;
  }

  .table-container {
      display: flex;
      justify-content: space-around;
      flex-wrap: wrap;
  }
  
  .image-container {
    display: flex;
    justify-content: space-between;
  }

  .image-column {
      flex: 1;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
  }

  .image-column img {
      max-width: 100%;
      height: auto;
  }

  .image-column p {
      text-align: center;
  }

  
</style>
</head>

<div style="height: 90pt;"></div>

<div class="header" style="border-bottom: 2px solid #a9b4c2; border-radius: 20px 20px 20px 20px; margin-top:20px; text-align: center; align-items: center; justify-content: center; margin-bottom:-5px; width: 100%;">
    <div>
        <h1 style="color: #2b3469; font-weight: 700; font-size:30px;">TransfQMix: Transformers for Leveraging the Graph Structure of Multi-Agent Reinforcement Learning Problems</h1>
        <h2 style="color: #2C4A67; margin-top: 0pt; margin-bottom: 0.2em;">Matteo Gallici, Mario Martín, Ivan Masmitja</h2>
        <h3 style="color:  #5763a6; margin-top: -2pt;">KEMLG Research Group, Universitat Politècnica de Catalunya, and Institut de Ciències del Mar (ICM-CSIC)</h3>
        <h3 style="color:  #5763a6; margin-top: 0pt;">AAMAS 20223</h3>
    </div>
</div>

--split--

<div class="header" style="background-color: #2b3469; border-color:#2b3469;">
  <h1 style="color: #e1e5fb;">Motivation</h1>
</div>


<div class="container">
  <ul style="margin-top:0px">
      <li>State-of-the-art approaches in MARL (MADDPG, VDN, QTran, QMix, QPlex, MAPPO, DICG) <b>focus on the learning model.</b></li>
      <li>No looking into observations and states: concatenations of features fed to NNs.</li>
      <li>Where do these features come from?</li>
  </ul>
  
  <div class="section" style="margin-top:-40px">
    <div class="section-content">
      <h2>Information Channels</h2>
      <ul>
        <li>Information usually comes from multiple sources: <i>observed entities, different sensors, communication channels.</i></li>
        <li>Agents can differentiate between information channels, because they occupy always the <b>same vector positions</b>.</li>
      </ul>
    </div>
    <div class="section-image">
      <img src="../images/trad_obs_vector.svg">
    </div>
  </div>

  <div class="section" style="margin-top:-40px">
    <div class="section-content">
      <h2>Entities Features</h2>
      <ul>
        <li>Often, a subset of <b>identical</b> features describes different observed entities.</li>
        <li><b>This is a graph structure!</b></li>
        <li>Self-Attention can be used to learn the edges of the latent graph <a href="https://arxiv.org/abs/2006.11438">Li et al., 2021</a></li>
      </ul>
    </div>
    <div class="section-image">
      <img src="../images/entity_obs_vector.svg">
    </div>
  </div>
</div>

<div class="header" style="background-color: #2b3469; border-color:#a9b4c2;">
  <h1 style="color: #e1e5fb;">Observation and State Matrices</h1>
</div>

<div class="container" style="background-color: #2b3469; color: #e1e5fb; border-color:#a9b4c2">
  <ul style="margin-top: -25px">
      <li>We reformulate the entire MARL problem in terms of graphs.</li>
      <li>The inputs to NNs are well-structured vertex features.</li>
      <li>The goal of the NNs is to learn the latent coordination graph.</li>
  </ul>
  <div class="section">
    <div class="section-content">
      <p>
        $$
        \begin{eqnarray}
        \mathbf{O}_t^a = 
        \begin{bmatrix}
        ent_1
        \\ \vdots \\ 
        ent_k
        \end{bmatrix}_t^a =
        \begin{bmatrix}
        f_{1,1} & \cdots & f_{1,z} \\
        \vdots & \ddots & \vdots \\ 
        f_{k,1} & \cdots & f_{k,z}
        \end{bmatrix}
        _t^a
        \end{eqnarray}
        $$
      </p>
    </div>
    <div class="section-content">
      <p>
        $$
        \begin{eqnarray}
        \mathbf{S}_t = 
        \begin{bmatrix}
        ent_1
        \\ \vdots \\ 
        ent_k
        \end{bmatrix}_t = 
        \begin{bmatrix}
        f_{1,1} & \cdots & f_{1,z} \\
        \vdots & \ddots & \vdots \\ 
        f_{k,1} & \cdots & f_{k,z}
        \end{bmatrix}_t
        \end{eqnarray}
        $$
      </p>
    </div>
  </div>
  <div class="section" style="margin-left: 50px; margin-right: 50px;">
    <table class="styled-table" style="color:#2b3469">
      <thead>
        <tr style="background-color:#e6e9fc; color:#2b3469">
          <th>Graph-Approach Advantages</th>
          <th>Disadvantages</th>
        </tr>
      </thead>
      <tbody>
        <tr style="background-color:white">
          <td style="padding: 5px">1. Better represents coordination problems.</td>
          <td style="padding: 5px">1. Cannot differentiate entities a-priori in observations, for instance, raw images (although we can for states).</td>
        </tr>
        <tr style="background-color:#f3f3f3">
          <td style="padding: 5px">2. Allow use of more appropriate NNs (GNN, Transformers).</td>
          <td style="padding: 5px">2. Cannot directly include last action and one hot encoding of agent id in the observation.</td>
        </tr>
        <tr style="background-color:white">
          <td style="padding: 5px">3. Makes the NNs parameters <b>invariant to the number of agents</b>.</td>
          <td style="padding: 5px"></td>
        </tr>
        <tr style="background-color:#f3f3f3">
          <td style="padding: 5px">4. Makes transfer learning and curriculum learning easy to implement.</td>
          <td style="padding: 5px"></td>
        </tr>
      </tbody>
  </table>
</div>

<div class="section">
<div class="section-content">
<ul>
  <li>
    Some information get lost when dropping concatenation:
    <ul>
      <li>Which of the vertices represent "me"?</li>
      <li>Which of the vertices are "collaborative agents"?</li>
    </ul>
  </li>
  <li>Easier solutions are flags into the vertex features.</li>
</ul>
</div>
<div class="section-content">
      <p>
        $$
        \begin{eqnarray}        
        f_{i,\texttt{IS_SELF}}^a =         \begin{cases}            1, & \text{if } i = a\\            0, & \text{otherwise.}        \end{cases}
        \end{eqnarray}
        $$
      </p>
      <p>
        $$
        \begin{eqnarray}        
        f_{i, \texttt{IS_AGENT}}^a =         
        \begin{cases}            1, & \text{if } i \in A\\            0, & \text{otherwise.}        \end{cases}    
        \end{eqnarray}
        $$
      </p>
</div>
</div>
</div>



<div class="header">
  <h1>Model</h1>
</div>

<div class="container">
  <div class="section-image" style="margin-top:-25px">
    <img src="../images/transf_qmix.svg">
  </div>
  <div class="section" style="align-items: flex-start; margin-top:-25px;">
    <div class="section-content">
      <h2>Transformer Mixer</h2>
      <p>Transformer hypernetwork:</p>
      <ul>
        <li>Agents hidden states (\(W_1\)): q-values embedding.</li>
        <li>Recurrent mechanism: (\(b_1, W_2, b_2\)): projection over \(Q_{tot}\).</li>
        <li>State embedder: environment reasoning.</li>
      </ul>
    </div>
    <div class="section-content">
      <h2>Transformer Agent</h2>
      <p>Similar architecture to UPDET, but:</p>
      <ul>
        <li>Q-Values are derived from the agent hidden state, reinforcing recurrent passing of gradient.</li>
        <li>Separate output layer used for Policy Decoupling.</li>
      </ul>
    </div>
  </div>

</div>

  <!--
  <div class="section">
    <div class="section-image">
      <img src="../images/transf_agent.svg">
    </div>
    <div class="section-content">
      <h2>Transformer Agent</h2>
      <p>Similar architecture to UPDET, but:</p>
      <ul>
        <li>Q-Values are derived from the agent hidden state, reinforcing recurrent passing of gradient.</li>
        <li>Separate output layer used for Policy Decoupling.</li>
      </ul>
    </div>
  </div>
  <div class="section">
    <div class="section-image">
      <img src="../images/transf_mixer.svg">
    </div>
    <div class="section-content">
      <h2>Transformer Mixer</h2>
      <p>Transformer hypernetwork:</p>
      <ul>
        <li>Agents hidden states:
          <ul>
            <li>\(W_1\): q-values embedding</li>
          </ul>
        </li>
        <li>Recurrent mechanism:
          <ul>
            <li>\(b1, W2, b2\): projection over \(Q_{tot}\)</li>
          </ul>
        </li>
        <li>State embedder: environment reasoning</li>
        </li>
      </ul>
    </div>
  -->


--split--

<div class="header" style="background-color: #0000;">
  <h1>Experiments</h1>
</div>

<div class="container" style="background-color: #0000;">
  <h2 style="margin-top:-10px">Spread</h2>
  <div class="image-container">
      <div class="image-column">
          <img src="../images/spread/spread_3v3.svg">
          <p>Spread 3v3</p>
      </div>
      <div class="image-column">
          <img src="../images/spread/spread_4v4.svg">
          <p>Spread 4v4</p>
      </div>
      <div class="image-column">
          <img src="../images/spread/spread_5v5.svg">
          <p>Spread 5v5</p>
      </div>
      <div class="image-column">
          <img src="../images/spread/spread_6v6.svg">
          <p>Spread 6v6</p>
      </div>
  </div>
  <h2 style="margin-top: 5px">StarCraft 2</h2>
  <div class="image-container">
      <div class="image-column">
          <img src="../images/sc2/5m_vs_6m.svg">
          <p>5m_vs_6m</p>
      </div>
      <div class="image-column">
          <img src="../images/sc2/8m_vs_9m.svg">
          <p>8m_vs_9m</p>
      </div>
      <div class="image-column">
          <img src="../images/sc2/27m_vs_30m.svg">
          <p>27m_vs_30m</p>
      </div>
      <div class="image-column">
          <img src="../images/sc2/6h_vs_8z.svg">
          <p>6h_vs_8z</p>
      </div>
  </div>
  <div class="image-container">
      <div class="image-column">
          <img src="../images/sc2/5s10z.svg">
          <p>5s10z</p>
      </div>
      <div class="image-column">
          <img src="../images/sc2/3s5z_vs_3s6z.svg">
          <p>3s5z_vs_3s6z</p>
      </div>
      <div class="image-column">
          <img src="../images/sc2/MMM2.svg">
          <p>MMM2</p>
      </div>
      <div class="image-column">
          <img src="../images/sc2/corridor.svg">
          <p>corridor</p>
      </div>
  </div>
</div>

<div class="header" style="background-color: #0000;">
  <h1 style="color: #2b3469;">Transfer Learning</h1>
</div>



<div class="container" style="background-color: #0000">
  <ul style="margin-top: -20px">
    <li>TransfQMix parameters are inviariant in respect to the number of entities.</li>
    <li>This enables transfer learning, curriculum learning and zero-shot transfer.</li>
  </ul>
  <h2 style="margin-top: 0; text-align: center">Models' Parameters Number</h2>
  <div class="section">
    <div class="section-content">
      <div class="table-wrapper">
        <table class="styled-table">
            <thead>
                <tr>
                    <th>Model</th>
                    <th>Agent</th>
                    <th>Mixer</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>TransfQMix</td>
                    <td>50k</td>
                    <td>50k</td>
                </tr>
                <tr>
                    <td>QMix</td>
                    <td>27k</td>
                    <td>18k</td>
                </tr>
                <tr>
                    <td>QPlex</td>
                    <td>27k</td>
                    <td>251k</td>
                </tr>
                <tr>
                    <td>O-CWQMix</td>
                    <td>27k</td>
                    <td>179k</td>
                </tr>
            </tbody>
        </table>
        <p style="text-align: center">Spread 3v3</p>
      </div>
    </div>
    <div class="section-content">
      <div class="table-wrapper">
        <table class="styled-table">
            <thead>
                <tr>
                    <th>Model</th>
                    <th>Agent</th>
                    <th>Mixer</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>TransfQMix</td>
                    <td>50k</td>
                    <td>50k</td>
                </tr>
                <tr>
                    <td>QMix</td>
                    <td>28k</td>
                    <td>56k</td>
                </tr>
                <tr>
                    <td>QPlex</td>
                    <td>28k</td>
                    <td>597k</td>
                </tr>
                <tr>
                    <td>O-CWQMix</td>
                    <td>28k</td>
                    <td>301k</td>
                </tr>
            </tbody>
        </table>
        <p style="text-align: center">Spread 6v6</p>
      </div>
    </div>
    <div class="section-content">
      <div class="table-wrapper">
        <table class="styled-table">
            <thead>
                <tr>
                    <th>Model</th>
                    <th>Agent</th>
                    <th>Mixer</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>TransfQMix</td>
                    <td>50k</td>
                    <td>50k</td>
                </tr>
                <tr>
                    <td>QMix</td>
                    <td>49k</td>
                    <td>283k</td>
                </tr>
                <tr>
                    <td>QPlex</td>
                    <td>49k</td>
                    <td>3184k</td>
                </tr>
                <tr>
                    <td>O-CWQMix</td>
                    <td>49k</td>
                    <td>1021k</td>
                </tr>
            </tbody>
        </table>
        <p style="text-align: center">SC2 27m_vs_30m</p>
      </div>
    </div>
  </div>
  <h2 style="margin-top: 0; text-align: center">Experiments</h2>
  <ul>
    <li>The learned policy is transferable between different team of agents.</li>
    <li><strong>Learning can be speeded up by transferring a learned policy.</strong></li>
  </ul>
  <div class="section" style="align-items: flex-end;">
    <div class="table-wrapper" style="width: 45%">
      <table class="styled-table" style="box-shadow: 0 0 10px rgba(0, 0, 0, 0.15); font-size: 15px;">
          <thead>
              <tr>
                  <th>Model</th>
                  <th>3v3</th>
                  <th>4v4</th>
                  <th>5v5</th>
                  <th>6v6</th>
              </tr>
          </thead>
          <tbody>
              <tr>
                  <td>TransfQMix (3v3)</td>
                  <td><strong>0.98</strong></td>
                  <td>0.88</td>
                  <td>0.8</td>
                  <td>0.75</td>
              </tr>
              <tr>
                  <td>TransfQMix (4v4)</td>
                  <td>0.96</td>
                  <td><strong>0.93</strong></td>
                  <td><strong>0.9</strong></td>
                  <td>0.86</td>
              </tr>
              <tr>
                  <td>TransfQMix (5v5)</td>
                  <td>0.88</td>
                  <td>0.85</td>
                  <td>0.82</td>
                  <td>0.82</td>
              </tr>
              <tr>
                  <td>TransfQMix (6v6)</td>
                  <td>0.91</td>
                  <td>0.88</td>
                  <td>0.85</td>
                  <td>0.84</td>
              </tr>
              <tr>
                  <td>TransfQMix (CL)</td>
                  <td>0.88</td>
                  <td>0.88</td>
                  <td>0.87</td>
                  <td><strong>0.87</strong></td>
              </tr>
              <tr>
                  <td>State-of-the-art</td>
                  <td>0.76</td>
                  <td>0.45</td>
                  <td>0.36</td>
                  <td>0.33</td>
              </tr>
          </tbody>
        </table>
        <p style="text-align: center">Spread: Zero Shot Transfer (POL)</p>
      </div>
      <div class="image-container" style="width: 55%">
        <div class="image-column" style="width: 22%">
            <img src="../images/sc2/8m_vs_9m to 5m_vs_6m.svg">
            <p style="text-align: center">SC2: 8m_vs_9m to 5m_vs_6m</p>
        </div>
        <div class="image-column" style="width: 22%">
            <img src="../images/sc2/5s10z to 3s5z_vs_3s6z.svg">
            <p style="text-align: center">SC2: 5s10z to 3s5z_vs_3s6z</p>
        </div>
      </div>
    </div>
</div>

<div class="header" style="text-align: center; background-color: #2b3469; border-color: #2b3469; color: #e1e5fb; padding-top:50px; margin-bottom: 0;">
  <img src="../images/github_qr.svg" style="width:250px" alt="Github QR Code">
  <p><a href="https://github.com/mttga/pymarl_transformers" target="_blank" style="color:#e1e5fb">github.com/mttga/pymarl_transformers</a></p>
</div>

<div class="container" style="text-align: center; background-color: #e6e9fc; padding-top:30px; padding-bottom:47px">
  <div class="section-image"> 
      <div class="image-column" style="margin:10px">
        <img src="../images/Logo_UPC.svg" style="max-width:55px;">
      </div>
      <div class="image-column" style="margin:10px">
        <img src="../images/BSC-blue.svg">
      </div>
      <div class="image-column" style="margin:10px">
        <img src="../images/logo-icm-color.svg"">
      </div>
      <div class="image-column" style="margin:10px">
        <img src="../images/csic_logo.svg" style="max-width: 120px;">
      </div>
      <div class="image-column" style="margin:10px">
        <img src="../images/logo-so-color.svg">
      </div>
    </div>
  </div>
</div>