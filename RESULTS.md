# Results and Models

Below, we provide the results for different ViT sizes (T for Tiny, S for Small, B for Base and L for Large) and different G2TM hyperparameters, on the ImageNet-1k dataset. We use the following notations to provide the values of G2TM's hyperparameters: G2TM@*L*[*tau*], where *L* is the layer where G2TM is applied and *tau* is the threshold used for inference.

**NOTE:** The results for the models' throughput and number of operations are respectively the median and the mean across the validation set using a batch size of 1.

**WARNING:** We are still looking for a solution to host all model checkpoints, in the meantime do not hesitate to request the checkpoints by contacting one of the authors.

## ImageNet (384×384)

The models are trained and evaluated on the ADE20K dataset with a crop resolution of 384×384. We provide two different settings for our G2TM module: one applied at the 1st layer with a threshold equal to 0.95 (noted G2TM@1[0.95]) and one applied at the 2nd layer with a threshold equal to 0.86 (noted G2TM@2[0.86]). The sign †, alongside the G2TM settings (e.g.: G2TM†@2[0.86]) indicates the use of the threshold curriculum training strategy, otherwise the threshold value is constant during training.

<table>
    <tr>
        <th>Model</th>
        <th>G2TM settings</th>
        <th>mIoU</th>
        <th>Im/sec</th>
        <th>GFLOPs</th>
        <th colspan="3">Download</th>
    </tr>
    <tr>
        <td>ViT-T/16</td>
        <td>-</td>
        <td><strong>78.0</strong></td>
        <td><strong>226.5</strong></td>
        <td>4.70</td>
        <td><a href="#setr-naive">model</a></td>
        <td><a href="#setr-naive">config</a></td>
        <td><a href="#setr-naive">log</a></td>
    </tr>
    <tr>
        <td>ViT-T/16</td>
        <td>G2TM@1[0.95]</td>
        <td>77.8</td>
        <td>193.0</td>
        <td>4.27</td>
        <td><a href="#setr-naive">model</a></td>
        <td><a href="#setr-naive">config</a></td>
        <td><a href="#setr-naive">log</a></td>
    </tr>
    <tr>
        <td>ViT-T/16</td>
        <td>G2TM†@2[0.86]</td>
        <td>77.6</td>
        <td>183.5</td>
        <td><strong>3.46</strong></td>
        <td><a href="#setr-naive">model</a></td>
        <td><a href="#setr-naive">config</a></td>
        <td><a href="#setr-naive">log</a></td>
    </tr>
    <tr>
        <td>ViT-S/16</td>
        <td>-</td>
        <td><strong>83.0</strong></td>
        <td><strong>195.5</strong></td>
        <td>15.52</td>
        <td><a href="#setr-naive">model</a></td>
        <td><a href="#setr-naive">config</a></td>
        <td><a href="#setr-naive">log</a></td>
    </tr>
    <tr>
        <td>ViT-S/16</td>
        <td>G2TM@1[0.95]</td>
        <td>82.9</td>
        <td>179.5</td>
        <td>14.98</td>
        <td><a href="#setr-naive">model</a></td>
        <td><a href="#setr-naive">config</a></td>
        <td><a href="#setr-naive">log</a></td>
    </tr>
    <tr>
        <td>ViT-S/16</td>
        <td>G2TM†@2[0.86]</td>
        <td>82.8</td>
        <td>180.6</td>
        <td>12.07</td>
        <td><a href="#setr-naive">model</a></td>
        <td><a href="#setr-naive">config</a></td>
        <td><a href="#setr-naive">log</a></td>
    </tr>
    <tr>
        <td>ViT-B/16</td>
        <td>-</td>
        <td><strong>84.8</strong></td>
        <td>88.0</td>
        <td>55.54</td>
        <td><a href="#setr-naive">model</a></td>
        <td><a href="#setr-naive">config</a></td>
        <td><a href="#setr-naive">log</a></td>
    </tr>
    <tr>
        <td>ViT-B/16</td>
        <td>G2TM@1[0.95]</td>
        <td><strong>84.8</strong></td>
        <td>101.6</td>
        <td>49.44</td>
        <td><a href="#setr-naive">model</a></td>
        <td><a href="#setr-naive">config</a></td>
        <td><a href="#setr-naive">log</a></td>
    </tr>
    <tr>
        <td>ViT-B/16</td>
        <td>G2TM†@2[0.86]</td>
        <td>84.5</td>
        <td><strong>103.6</strong></td>
        <td><strong>41.05</strong></td>
        <td><a href="#setr-naive">model</a></td>
        <td><a href="#setr-naive">config</a></td>
        <td><a href="#setr-naive">log</a></td>
    </tr>
    <tr>
        <td>ViT-L/16</td>
        <td>-</td>
        <td><strong>86.2</strong></td>
        <td>31.6</td>
        <td>191.21</td>
        <td><a href="#setr-naive">model</a></td>
        <td><a href="#setr-naive">config</a></td>
        <td><a href="#setr-naive">log</a></td>
    </tr>
    <tr>
        <td>ViT-L/16</td>
        <td>G2TM@1[0.95]</td>
        <td>86.0</td>
        <td>32.6</td>
        <td>174.40</td>
        <td><a href="#setr-naive">model</a></td>
        <td><a href="#setr-naive">config</a></td>
        <td><a href="#setr-naive">log</a></td>
    </tr>
    <tr>
        <td>ViT-L/16</td>
        <td>G2TM@2[0.86]</td>
        <td>85.5</td>
        <td><strong>44.8</strong></td>
        <td><strong>124.64</strong></td>
        <td><a href="#setr-naive">model</a></td>
        <td><a href="#setr-naive">config</a></td>
        <td><a href="#setr-naive">log</a></td>
    </tr>
</table>