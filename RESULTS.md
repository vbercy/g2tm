# Results and Models

Below, we provide the results for both decoders (Linear and Mask Transformer) and different backbone sizes (T for Tiny, S for Small, B for Base and L for Large), as well as for different G2TM hyperparameters and datasets. We use the following notations to provide the values of G2TM's hyperparameters: G2TM@*L*[*tau*], where *L* is the layer where G2TM is applied and *tau* is the threshold used for inference.

**NOTE:** The results for the models' throughput and number of operations are respectively the median and the mean across the validation set using a batch size of 1.

**WARNING:** We are still looking for a solution to host all model checkpoints, in the meantime do not hesitate to request the checkpoints by contacting one of the authors.

## ADE20K (512×512)

The models are trained and evaluated on the ADE20K dataset with a crop resolution of 512×512. We provide two different settings for our G2TM module: one applied at the 1st layer with a threshold equal to 0.95 (noted G2TM@1[0.95]) and one applied at the 2nd layer with a threshold equal to 0.88 (noted G2TM@2[0.88]). The sign †, alongside the G2TM settings (e.g.: G2TM†@2[0.88]) indicates the use of the threshold curriculum training strategy, otherwise the threshold value is constant during training.

### Linear decoder

The following models use a Linear Decoder following the ViT backbone.

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
        <td>ViT-T/16</td>
        <td>-</td>
        <td><strong>40.1</strong></td>
        <td><strong>174.1</strong></td>
        <td>10.65</td>
        <td><a href="#linear-decoder">model</a></td>
        <td><a href="#linear-decoder">config</a></td>
        <td><a href="#linear-decoder">log</a></td>
    </tr>
    <tr>
        <td>ViT-T/16</td>
        <td>G2TM@1[0.95]</td>
        <td>39.1</td>
        <td>164.1</td>
        <td>7.06</td>
        <td><a href="#linear-decoder">model</a></td>
        <td><a href="#linear-decoder">config</a></td>
        <td><a href="#linear-decoder">log</a></td>
    </tr>
    <tr>
        <td>ViT-T/16</td>
        <td>G2TM†@2[0.88]</td>
        <td>39.3</td>
        <td>160.7</td>
        <td><strong>6.57</strong></td>
        <td><a href="#linear-decoder">model</a></td>
        <td><a href="#linear-decoder">config</a></td>
        <td><a href="#linear-decoder">log</a></td>
    </tr>
    <tr>
        <td>ViT-S/16</td>
        <td>-</td>
        <td>46.2</td>
        <td>114.5</td>
        <td>32.02</td>
        <td><a href="#linear-decoder">model</a></td>
        <td><a href="#linear-decoder">config</a></td>
        <td><a href="#linear-decoder">log</a></td>
    </tr>
    <tr>
        <td>ViT-S/16</td>
        <td>G2TM@1[0.95]</td>
        <td><strong>46.3</strong></td>
        <td>123.4</td>
        <td>24.37</td>
        <td><a href="#linear-decoder">model</a></td>
        <td><a href="#linear-decoder">config</a></td>
        <td><a href="#linear-decoder">log</a></td>
    </tr>
    <tr>
        <td>ViT-S/16</td>
        <td>G2TM†@2[0.88]</td>
        <td>46.1</td>
        <td><strong>131.2</strong></td>
        <td><strong>22.55</strong></td>
        <td><a href="#linear-decoder">model</a></td>
        <td><a href="#linear-decoder">config</a></td>
        <td><a href="#linear-decoder">log</a></td>
    </tr>
    <tr>
        <td>ViT-B/16</td>
        <td>-</td>
        <td><strong>48.4</strong></td>
        <td>53.5</td>
        <td>107.4</td>
        <td><a href="#linear-decoder">model</a></td>
        <td><a href="#linear-decoder">config</a></td>
        <td><a href="#linear-decoder">log</a></td>
    </tr>
    <tr>
        <td>ViT-B/16</td>
        <td>G2TM@1[0.95]</td>
        <td>44.5</td>
        <td><strong>74.3</strong></td>
        <td><strong>58.92</strong></td>
        <td><a href="#linear-decoder">model</a></td>
        <td><a href="#linear-decoder">config</a></td>
        <td><a href="#linear-decoder">log</a></td>
    </tr>
    <tr>
        <td>ViT-B/16</td>
        <td>G2TM†@2[0.88]</td>
        <td>47.3</td>
        <td>70.0</td>
        <td>67.54</td>
        <td><a href="#linear-decoder">model</a></td>
        <td><a href="#linear-decoder">config</a></td>
        <td><a href="#linear-decoder">log</a></td>
    </tr>
    <tr>
        <td>ViT-L/16</td>
        <td>-</td>
        <td><strong>51.3</strong></td>
        <td>16.5</td>
        <td>362.56</td>
        <td><a href="#linear-decoder">model</a></td>
        <td><a href="#linear-decoder">config</a></td>
        <td><a href="#linear-decoder">log</a></td>
    </tr>
    <tr>
        <td>ViT-L/16</td>
        <td>G2TM@1[0.95]</td>
        <td>50.7</td>
        <td>23.8</td>
        <td>253.92</td>
        <td><a href="#linear-decoder">model</a></td>
        <td><a href="#linear-decoder">config</a></td>
        <td><a href="#linear-decoder">log</a></td>
    </tr>
    <tr>
        <td>ViT-L/16</td>
        <td>G2TM†@2[0.88]</td>
        <td>49.2</td>
        <td><strong>29.6</strong></td>
        <td><strong>7.84</strong></td>
        <td><a href="#linear-decoder">model</a></td>
        <td><a href="#linear-decoder">config</a></td>
        <td><a href="#linear-decoder">log</a></td>
    </tr>
</table>

### Mask Transformer decoder

The following models use a Mask Transformer Decoder following the ViT backbone.

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
        <td>ViT-T/16</td>
        <td>-</td>
        <td><strong>40.7</strong></td>
        <td><strong>147.4</strong></td>
        <td>12.83</td>
        <td><a href="#mask-transformer-decoder">model</a></td>
        <td><a href="#mask-transformer-decoder">config</a></td>
        <td><a href="#mask-transformer-decoder">log</a></td>
    </tr>
    <tr>
        <td>ViT-T/16</td>
        <td>G2TM@1[0.95]</td>
        <td>39.1</td>
        <td>141.2</td>
        <td>8.55</td>
        <td><a href="#mask-transformer-decoder">model</a></td>
        <td><a href="#mask-transformer-decoder">config</a></td>
        <td><a href="#mask-transformer-decoder">log</a></td>
    </tr>
    <tr>
        <td>ViT-T/16</td>
        <td>G2TM†@2[0.88]</td>
        <td>39.9</td>
        <td>141.3</td>
        <td><strong>7.84</strong></td>
        <td><a href="#mask-transformer-decoder">model</a></td>
        <td><a href="#mask-transformer-decoder">config</a></td>
        <td><a href="#mask-transformer-decoder">log</a></td>
    </tr>
    <tr>
        <td>ViT-S/16</td>
        <td>-</td>
        <td>46.5</td>
        <td>94.8</td>
        <td>38.62</td>
        <td><a href="#mask-transformer-decoder">model</a></td>
        <td><a href="#mask-transformer-decoder">config</a></td>
        <td><a href="#mask-transformer-decoder">log</a></td>
    </tr>
    <tr>
        <td>ViT-S/16</td>
        <td>G2TM@1[0.95]</td>
        <td>46.1</td>
        <td>106.5</td>
        <td>29.28</td>
        <td><a href="#mask-transformer-decoder">model</a></td>
        <td><a href="#mask-transformer-decoder">config</a></td>
        <td><a href="#mask-transformer-decoder">log</a></td>
    </tr>
    <tr>
        <td>ViT-S/16</td>
        <td>G2TM†@2[0.88]</td>
        <td><strong>46.6</strong></td>
        <td><strong>113.4</strong></td>
        <td><strong>26.15</strong></td>
        <td><a href="#mask-transformer-decoder">model</a></td>
        <td><a href="#mask-transformer-decoder">config</a></td>
        <td><a href="#mask-transformer-decoder">log</a></td>
    </tr>
    <tr>
        <td>ViT-B/16</td>
        <td>-</td>
        <td><strong>49.6</strong></td>
        <td>43.2</td>
        <td>129.57</td>
        <td><a href="#mask-transformer-decoder">model</a></td>
        <td><a href="#mask-transformer-decoder">config</a></td>
        <td><a href="#mask-transformer-decoder">log</a></td>
    </tr>
    <tr>
        <td>ViT-B/16</td>
        <td>G2TM@1[0.95]</td>
        <td>46.5</td>
        <td><strong>62.2</strong></td>
        <td><strong>75.05</strong></td>
        <td><a href="#mask-transformer-decoder">model</a></td>
        <td><a href="#mask-transformer-decoder">config</a></td>
        <td><a href="#mask-transformer-decoder">log</a></td>
    </tr>
    <tr>
        <td>ViT-B/16</td>
        <td>G2TM†@2[0.88]</td>
        <td>48.7</td>
        <td>58.5</td>
        <td>81.15</td>
        <td><a href="#mask-transformer-decoder">model</a></td>
        <td><a href="#mask-transformer-decoder">config</a></td>
        <td><a href="#mask-transformer-decoder">log</a></td>
    </tr>
    <tr>
        <td>ViT-L/16</td>
        <td>-</td>
        <td><strong>52.3</strong></td>
        <td>14.9</td>
        <td>400.08</td>
        <td><a href="#mask-transformer-decoder">model</a></td>
        <td><a href="#mask-transformer-decoder">config</a></td>
        <td><a href="#mask-transformer-decoder">log</a></td>
    </tr>
    <tr>
        <td>ViT-L/16</td>
        <td>G2TM@1[0.95]</td>
        <td>52.0</td>
        <td>20.4</td>
        <td>284.58</td>
        <td><a href="#mask-transformer-decoder">model</a></td>
        <td><a href="#mask-transformer-decoder">config</a></td>
        <td><a href="#mask-transformer-decoder">log</a></td>
    </tr>
    <tr>
        <td>ViT-L/16</td>
        <td>G2TM†@2[0.88]</td>
        <td>50.3</td>
        <td><strong>26.0</strong></td>
        <td><strong>210.96</strong></td>
        <td><a href="#mask-transformer-decoder">model</a></td>
        <td><a href="#mask-transformer-decoder">config</a></td>
        <td><a href="#mask-transformer-decoder">log</a></td>
    </tr>
</table>

<h2 id="citysmall">Cityscapes (768×768)</h2>

The models use a Mask Transformer decoder following a ViT backbone and they are trained and evaluated on the Cityscapes dataset with a crop resolution of 768×768. We provide only one setting for our G2TM module: applied at the 2nd layer with a threshold equal to 0.95 (noted G2TM@1[0.95]). The sign †, alongside the G2TM settings (e.g.: G2TM†@2[0.95]) indicates the use of the threshold curriculum training strategy, otherwise the threshold value is constant during training.

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
        <td>ViT-T/16</td>
        <td>-</td>
        <td><strong>73.5</strong></td>
        <td>64.7</td>
        <td>43.55</td>
        <td><a href="#citysmall">model</a></td>
        <td><a href="#citysmall">config</a></td>
        <td><a href="#citysmall">log</a></td>
    </tr>
    <tr>
        <td>ViT-T/16</td>
        <td>G2TM†@2[0.95]</td>
        <td>72.0</td>
        <td><strong>80.1</strong></td>
        <td><strong>29.66</strong></td>
        <td><a href="#citysmall">model</a></td>
        <td><a href="#citysmall">config</a></td>
        <td><a href="#citysmall">log</a></td>
    </tr>
    <tr>
        <td>ViT-S/16</td>
        <td>-</td>
        <td><strong>76.6</strong></td>
        <td>31.5</td>
        <td>115.98</td>
        <td><a href="#citysmall">model</a></td>
        <td><a href="#citysmall">config</a></td>
        <td><a href="#citysmall">log</a></td>
    </tr>
    <tr>
        <td>ViT-S/16</td>
        <td>G2TM†@2[0.95]</td>
        <td>76.2</td>
        <td><strong>48.7</strong></td>
        <td><strong>84.04</strong></td>
        <td><a href="#citysmall">model</a></td>
        <td><a href="#citysmall">config</a></td>
        <td><a href="#citysmall">log</a></td>
    </tr>
    <tr>
        <td>ViT-B/16</td>
        <td>-</td>
        <td>77.6</td>
        <td>14.4</td>
        <td>347.60</td>
        <td><a href="#citysmall">model</a></td>
        <td><a href="#citysmall">config</a></td>
        <td><a href="#citysmall">log</a></td>
    </tr>
    <tr>
        <td>ViT-B/16</td>
        <td>G2TM†@2[0.95]</td>
        <td>76.0</td>
        <td><strong>23.4</strong></td>
        <td><strong>230.47</strong></td>
        <td><a href="#citysmall">model</a></td>
        <td><a href="#citysmall">config</a></td>
        <td><a href="#citysmall">log</a></td>
    </tr>
    <tr>
        <td>ViT-L/16</td>
        <td>-</td>
        <td><strong>79.1</strong></td>
        <td>5.3</td>
        <td>1045.19</td>
        <td><a href="#citysmall">model</a></td>
        <td><a href="#citysmall">config</a></td>
        <td><a href="#citysmall">log</a></td>
    </tr>
    <tr>
        <td>ViT-L/16</td>
        <td>G2TM†@2[0.95]</td>
        <td>76.8</td>
        <td><strong>10.3</strong></td>
        <td><strong>599.24</strong></td>
        <td><a href="#citysmall">model</a></td>
        <td><a href="#citysmall">config</a></td>
        <td><a href="#citysmall">log</a></td>
    </tr>
</table>

<h2 id="citylarge">Cityscapes (1024×1024)</h2>

The models use a Mask Transformer decoder following a ViT backbone and they are trained and evaluated on the Cityscapes dataset with a crop resolution of 1024×1024. We provide only one setting for our G2TM module: applied at the 2nd layer with a threshold equal to 0.95 (noted G2TM@1[0.95]). The sign †, alongside the G2TM settings (e.g.: G2TM†@2[0.95]) indicates the use of the threshold curriculum training strategy, otherwise the threshold value is constant during training.

**NOTE:** We do not provide the results on a Segmenter-Large model due to GPU memory limitations.

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
        <td>ViT-T/16</td>
        <td>-</td>
        <td><strong>73.1</strong></td>
        <td>32.8</td>
        <td>116.86</td>
        <td><a href="#citylarge">model</a></td>
        <td><a href="#citylarge">config</a></td>
        <td><a href="#citylarge">log</a></td>
    </tr>
    <tr>
        <td>ViT-T/16</td>
        <td>G2TM†@2[0.95]</td>
        <td>71.9</td>
        <td><strong>42.8</strong></td>
        <td><strong>70.85</strong></td>
        <td><a href="#citylarge">model</a></td>
        <td><a href="#citylarge">config</a></td>
        <td><a href="#citylarge">log</a></td>
    </tr>
    <tr>
        <td>ViT-S/16</td>
        <td>-</td>
        <td><strong>76.6</strong></td>
        <td>15.7</td>
        <td>285.03</td>
        <td><a href="#citylarge">model</a></td>
        <td><a href="#citylarge">config</a></td>
        <td><a href="#citylarge">log</a></td>
    </tr>
    <tr>
        <td>ViT-S/16</td>
        <td>G2TM†@2[0.95]</td>
        <td>75.4</td>
        <td><strong>20.9</strong></td>
        <td><strong>186.59</strong></td>
        <td><a href="#citylarge">model</a></td>
        <td><a href="#citylarge">config</a></td>
        <td><a href="#citylarge">log</a></td>
    </tr>
    <tr>
        <td>ViT-B/16</td>
        <td>-</td>
        <td><strong>77.1</strong></td>
        <td>6.8</td>
        <td>775.51</td>
        <td><a href="#citylarge">model</a></td>
        <td><a href="#citylarge">config</a></td>
        <td><a href="#citylarge">log</a></td>
    </tr>
    <tr>
        <td>ViT-B/16</td>
        <td>G2TM†@2[0.95]</td>
        <td>75.2</td>
        <td><strong>11.7</strong></td>
        <td><strong>439.15</strong></td>
        <td><a href="#citylarge">model</a></td>
        <td><a href="#citylarge">config</a></td>
        <td><a href="#citylarge">log</a></td>
    </tr>
</table>
