# Results and Models

Below, we provide the results for the three decoders introduced in the [original SETR paper](https://arxiv.org/abs/2012.15840) and different backbone sizes (B for Base and L for Large), as well as for different G2TM hyperparameters, on the ADE20K dataset. We use the following notations to provide the values of G2TM's hyperparameters: G2TM@*L*[*tau*], where *L* is the layer where G2TM is applied and *tau* is the threshold used for inference.

**NOTE:** The results for the models' throughput and number of operations are respectively the median and the mean across the validation set using a batch size of 1.

**WARNING:** We are still looking for a solution to host all model checkpoints, in the meantime do not hesitate to request the checkpoints by contacting one of the authors.

## ADE20K (512×512)

The models are trained and evaluated on the ADE20K dataset with a crop resolution of 512×512. We provide two different settings for our G2TM module: one applied at the 1st layer with a threshold equal to 0.95 (noted G2TM@1[0.95]) and one applied at the 2nd layer with a threshold equal to 0.88 (noted G2TM@2[0.88]). The sign †, alongside the G2TM settings (e.g.: G2TM†@2[0.88]) indicates the use of the threshold curriculum training strategy, otherwise the threshold value is constant during training.

### SETR-Naive

The following models use the Naive Decoder from the orignal article, following the ViT backbone.

<table>
    <tr>
        <th>Backbone</th>
        <th>G2TM settings</th>
        <th>mIoU</th>
        <th>Im/sec</th>
        <th>GFLOPs</th>
        <th colspan="3">Download</th>
    </tr>
    <tr>
        <td>ViT-B/16</td>
        <td>-</td>
        <td><strong>48.8</strong></td>
        <td>53.5</td>
        <td>18.29</td>
        <td><a href="#setr-naive">model</a></td>
        <td><a href="#setr-naive">config</a></td>
        <td><a href="#setr-naive">log</a></td>
    </tr>
    <tr>
        <td>ViT-B/16</td>
        <td>G2TM@1[0.95]</td>
        <td>46.9</td>
        <td><strong>72.9</strong></td>
        <td><strong>62.47</strong></td>
        <td><a href="#setr-naive">model</a></td>
        <td><a href="#setr-naive">config</a></td>
        <td><a href="#setr-naive">log</a></td>
    </tr>
    <tr>
        <td>ViT-B/16</td>
        <td>G2TM†@2[0.88]</td>
        <td>47.9</td>
        <td>69.9</td>
        <td>68.29</td>
        <td><a href="#setr-naive">model</a></td>
        <td><a href="#setr-naive">config</a></td>
        <td><a href="#setr-naive">log</a></td>
    </tr>
    <tr>
        <td>ViT-L/16</td>
        <td>-</td>
        <td><strong>52.0</strong></td>
        <td>16.5</td>
        <td>363.47</td>
        <td><a href="#setr-naive">model</a></td>
        <td><a href="#setr-naive">config</a></td>
        <td><a href="#setr-naive">log</a></td>
    </tr>
    <tr>
        <td>ViT-L/16</td>
        <td>G2TM@1[0.95]</td>
        <td>51.8</td>
        <td>24.3</td>
        <td>250.14</td>
        <td><a href="#setr-naive">model</a></td>
        <td><a href="#setr-naive">config</a></td>
        <td><a href="#setr-naive">log</a></td>
    </tr>
    <tr>
        <td>ViT-L/16</td>
        <td>G2TM†@2[0.88]</td>
        <td>49.9</td>
        <td><strong>29.6</strong></td>
        <td><strong>197.61</strong></td>
        <td><a href="#setr-naive">model</a></td>
        <td><a href="#setr-naive">config</a></td>
        <td><a href="#setr-naive">log</a></td>
    </tr>
</table>

### SETR-PUP

The following models use the PUP decoder from the orignal article, following the ViT backbone.

<table>
    <tr>
        <th>Backbone</th>
        <th>G2TM settings</th>
        <th>mIoU</th>
        <th>Im/sec</th>
        <th>GFLOPs</th>
        <th colspan="3">Download</th>
    </tr>
    <tr>
        <td>ViT-B/16</td>
        <td>-</td>
        <td><strong>49.5</strong></td>
        <td>46.0</td>
        <td>170.30</td>
        <td><a href="#setr-pup">model</a></td>
        <td><a href="#setr-pup">config</a></td>
        <td><a href="#setr-pup">log</a></td>
    </tr>
    <tr>
        <td>ViT-B/16</td>
        <td>G2TM@1[0.95]</td>
        <td>47.5</td>
        <td><strong>61.5</strong></td>
        <td><strong>126.87</strong></td>
        <td><a href="#setr-pup">model</a></td>
        <td><a href="#setr-pup">config</a></td>
        <td><a href="#setr-pup">log</a></td>
    </tr>
    <tr>
        <td>ViT-B/16</td>
        <td>G2TM†@2[0.88]</td>
        <td>48.2</td>
        <td>59.7</td>
        <td>131.01</td>
        <td><a href="#setr-pup">model</a></td>
        <td><a href="#setr-pup">config</a></td>
        <td><a href="#setr-pup">log</a></td>
    </tr>
    <tr>
        <td>ViT-L/16</td>
        <td>-</td>
        <td><strong>52.3</strong></td>
        <td>14.9</td>
        <td>400.08</td>
        <td><a href="#setr-pup">model</a></td>
        <td><a href="#setr-pup">config</a></td>
        <td><a href="#setr-pup">log</a></td>
    </tr>
    <tr>
        <td>ViT-L/16</td>
        <td>G2TM@1[0.95]</td>
        <td>50.5</td>
        <td>23.7</td>
        <td>297.88</td>
        <td><a href="#setr-pup">model</a></td>
        <td><a href="#setr-pup">config</a></td>
        <td><a href="#setr-pup">log</a></td>
    </tr>
    <tr>
        <td>ViT-L/16</td>
        <td>G2TM@2[0.88]</td>
        <td>49.1</td>
        <td><strong>27.8</strong></td>
        <td><strong>256.11</strong></td>
        <td><a href="#setr-pup">model</a></td>
        <td><a href="#setr-pup">config</a></td>
        <td><a href="#setr-pup">log</a></td>
    </tr>
</table>

### SETR-MLA

The following models use the MLA decoder from the orignal article, following the ViT backbone.

<table>
    <tr>
        <th>Backbone</th>
        <th>G2TM settings</th>
        <th>mIoU</th>
        <th>Im/sec</th>
        <th>GFLOPs</th>
        <th colspan="3">Download</th>
    </tr>
    <tr>
        <td>ViT-B/16</td>
        <td>-</td>
        <td><strong>49.7</strong></td>
        <td>49.4</td>
        <td>133.86</td>
        <td><a href="#setr-mla">model</a></td>
        <td><a href="#setr-mla">config</a></td>
        <td><a href="#setr-mla">log</a></td>
    </tr>
    <tr>
        <td>ViT-B/16</td>
        <td>G2TM@1[0.95]</td>
        <td>48.1</td>
        <td><strong>64.4</strong></td>
        <td><strong>90.81</strong></td>
        <td><a href="#setr-mla">model</a></td>
        <td><a href="#setr-mla">config</a></td>
        <td><a href="#setr-mla">log</a></td>
    </tr>
    <tr>
        <td>ViT-B/16</td>
        <td>G2TM†@2[0.88]</td>
        <td>48.8</td>
        <td>62.4</td>
        <td>94.46</td>
        <td><a href="#setr-mla">model</a></td>
        <td><a href="#setr-mla">config</a></td>
        <td><a href="#setr-mla">log</a></td>
    </tr>
    <tr>
        <td>ViT-L/16</td>
        <td>-</td>
        <td><strong>51.8</strong></td>
        <td>16.0</td>
        <td>401.94</td>
        <td><a href="#setr-mla">model</a></td>
        <td><a href="#setr-mla">config</a></td>
        <td><a href="#setr-mla">log</a></td>
    </tr>
    <tr>
        <td>ViT-L/16</td>
        <td>G2TM@1[0.95]</td>
        <td>51.1</td>
        <td>22.7</td>
        <td>290.86</td>
        <td><a href="#setr-mla">model</a></td>
        <td><a href="#setr-mla">config</a></td>
        <td><a href="#setr-mla">log</a></td>
    </tr>
    <tr>
        <td>ViT-L/16</td>
        <td>G2TM†@2[0.88]</td>
        <td>50.2</td>
        <td><strong>27.8</strong></td>
        <td><strong>235.58</strong></td>
        <td><a href="#setr-mla">model</a></td>
        <td><a href="#setr-mla">config</a></td>
        <td><a href="#setr-mla">log</a></td>
    </tr>
</table>