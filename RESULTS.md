# Results and Models

Below, we provide the results for the EoMT-Large model for different G2TM hyperparameters and different pre-training recipe (AugReg and DINOv2), on the ADE20K dataset. We use the following notations to provide the values of G2TM's hyperparameters: G2TM@*L*[*tau*], where *L* is the layer where G2TM is applied and *tau* is the threshold used for inference.

**NOTE:** The results for the models' throughput and number of operations are respectively the median and the mean across the validation set using a batch size of 1.

**WARNING:** We are still looking for a solution to host all model checkpoints, in the meantime do not hesitate to request the checkpoints by contacting one of the authors.

## ADE20K (512×512)

The models are trained and evaluated on the ADE20K dataset with a crop resolution of 512×512. We provide two different settings for our G2TM module: one applied at the 1st layer with a threshold equal to 0.95 (noted G2TM@1[0.95]) and one applied at the 2nd layer with a threshold equal to 0.88 (noted G2TM@2[0.88]). The sign †, alongside the G2TM settings (e.g.: G2TM†@2[0.88]) indicates the use of the threshold curriculum training strategy, otherwise the threshold value is constant during training.

### AugReg

Here, the ViT-Large, the main component of the following EoMT-Large models, has been pre-trained on the following the AugReg (Augmentation and Regularization) methodology, using 16×16 patches.

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
        <td>EoMT-L/16</td>
        <td>-</td>
        <td><strong>52.3</strong></td>
        <td>60.9</td>
        <td>392.82</td>
        <td><a href="#augreg">model</a></td>
        <td><a href="#augreg">config</a></td>
        <td><a href="#augreg">log</a></td>
    </tr>
    <tr>
        <td>EoMT-L/16</td>
        <td>G2TM@1[0.95]</td>
        <td>52.0</td>
        <td>71.4</td>
        <td>297.10</td>
        <td><a href="#augreg">model</a></td>
        <td><a href="#augreg">config</a></td>
        <td><a href="#augreg">log</a></td>
    </tr>
    <tr>
        <td>EoMT-L/16</td>
        <td>G2TM†@2[0.88]</td>
        <td>52.2</td>
        <td><strong>77.8</strong></td>
        <td><strong>260.23</strong></td>
        <td><a href="#augreg">model</a></td>
        <td><a href="#augreg">config</a></td>
        <td><a href="#augreg">log</a></td>
    </tr>
</table>

### DINOv2

Here, the ViT-Large, the main component of the following EoMT-Large models, has been pre-trained on the following the DINOv2 methodology, using 14×14 patches. The models are then fine-tuned using 16×16 patches.

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
        <td>ViT-L/16</td>
        <td>-</td>
        <td><strong>58.1</strong></td>
        <td>60.9</td>
        <td>392.82</td>
        <td><a href="#dinov2">model</a></td>
        <td><a href="#dinov2">config</a></td>
        <td><a href="#dinov2">log</a></td>
    </tr>
    <tr>
        <td>ViT-L/16</td>
        <td>G2TM@1[0.95]</td>
        <td>52.5</td>
        <td>81.2</td>
        <td>231.71</td>
        <td><a href="#dinov2">model</a></td>
        <td><a href="#dinov2">config</a></td>
        <td><a href="#dinov2">log</a></td>
    </tr>
    <tr>
        <td>ViT-L/16</td>
        <td>G2TM@2[0.88]</td>
        <td>49.3</td>
        <td><strong>90.4</strong></td>
        <td><strong>151.66</strong></td>
        <td><a href="#dinov2">model</a></td>
        <td><a href="#dinov2">config</a></td>
        <td><a href="#dinov2">log</a></td>
    </tr>
    <tr>
        <td>ViT-L/16</td>
        <td>G2TM@2[0.88]</td>
        <td>56.7</td>
        <td>80.7</td>
        <td>228.25</td>
        <td><a href="#dinov2">model</a></td>
        <td><a href="#dinov2">config</a></td>
        <td><a href="#dinov2">log</a></td>
    </tr>
</table>
