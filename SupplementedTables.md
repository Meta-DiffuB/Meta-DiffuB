**Supplemented Table 1**: In DiffuSeq, the same maximum length, diffusion steps, model parameters, and training batch size are used across all four datasets mentioned in the paper. Consequently, our scheduler settings are also consistent across the four datasets, regardless of the increase in inference time, training time, and model parameters.
|  | increased parameters (%) | increased training time (%) | increased inference time (%) |
| :-- | :--: | :--: | :--: |
| Meta-DiffuB | 2.2% | 5% | 0.5% |

<br>
<br>

**Supplemented Table 2:** Meta-DiffuB on Machine Translation Datasets. When S2S-Diffusion models are used as the exploiter models in our Meta-Diffu$B$ framework and are trained from scratch, they achieve performance that surpasses their original results, as indicated in bold.
<table>
  <tr>
    <th rowspan="2">Methods</th>
    <th >IWSLT14 DE-EN</th>
    <th >WMT14 DE-EN</th>
  </tr>
  <tr>
    <th>SacreBLEU (↑)</th>
    <th>SacreBLEU (↑)</th>
  </tr>
  <tr>
    <td>DiffuSeq</td>
    <td>29.43</td>
    <td>22.72</td>
  </tr>
  <tr>
    <td>Meta-DiffuB (exploiter=DiffuSeq)</td>
    <td><strong>31.71</strong></td>
    <td><strong>26.17</strong></td>
  </tr>
  <tr>
    <td>SeqDiffuSeq</td>
    <td>30.16</td>
    <td>23.28</td>
  </tr>
  <tr>
    <td>Meta-DiffuB (exploiter=SeqDiffuSeq)</td>
    <td><strong>32.41</strong></td>
    <td><strong>26.14</strong></td>
  </tr>
  <tr>
    <td>Dinoiser</td>
    <td>31.61</td>
    <td>30.30</td>
  </tr>
  <tr>
    <td>Meta-DiffuB (exploiter=Dinoiser)</td>
    <td>3<strong>3.82</strong></td>
    <td><strong>32.09</strong></td>
  </tr>
</table>

<br>
<br>

**Supplemented Table 3**: More baselines compared with our Meta-DiffuB. DiffuSeq-V2 does not report the result on QG dataset. When S2S-Diffusion models are used as the exploiter models in our Meta-Diffu$B$ framework and are trained from scratch, they achieve performance that surpasses their original results, as indicated in bold.
<table>
  <tr>
    <th rowspan="2">Methods</th>
    <th colspan="2">QQP</th>
    <th colspan="2">QG</th>
  </tr>
  <tr>
    <th>BLEU (↑)</th>
    <th>BERTScore (↑)</th>
    <th>BLEU (↑)</th>
    <th>BERTScore (↑)</th>
  </tr>
  <tr>
    <td>DiffuSeq</td>
    <td>0.2413</td>
    <td>0.8365</td>
    <td>0.1731</td>
    <td>0.6123</td>
  </tr>
  <tr>
    <td>Meta-DiffuB (exploiter=DiffuSeq)</td>
    <td><strong>0.2552</strong></td>
    <td><strong>0.8821</strong></td>
    <td><strong>0.1826</strong></td>
    <td><strong>0.6357</strong></td>
  </tr>
  <tr>
    <td>DiffuSeq-v2 [1]</td>
    <td>0.2411</td>
    <td>0.8393</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>Meta-DiffuB (exploiter=DiffuSeq-v2)</td>
    <td><strong>0.2556</strong></td>
    <td><strong>0.8829</strong></td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>BG-DiffuSeq [2]</td>
    <td>0.2619</td>
    <td>0.8427</td>
    <td>0.1744</td>
    <td>0.628</td>
  </tr>
  <tr>
    <td>Meta-DiffuB (exploiter=BG-DiffuSeq)</td>
    <td><strong>0.2790</strong></td>
    <td><strong>0.8757</strong></td>
    <td><strong>0.1838</strong></td>
    <td><strong>0.6571</strong></td>
  </tr>
  <tr>
    <td>TESS [3]</td>
    <td>0.3020</td>
    <td>0.8570</td>
    <td>0.1950</td>
    <td>0.6580</td>
  </tr>
  <tr>
    <td>Meta-DiffuB (exploiter=TESS)</td>
    <td><strong>0.3142</strong></td>
    <td><strong>0.8975</strong></td>
    <td><strong>0.2055</strong></td>
    <td><strong>0.6761</strong></td>
  </tr>
</table>

[1] Diffuseq-v2: https://arxiv.org/pdf/2310.05793 <br>
[2] BG-DiffuSeq: https://arxiv.org/pdf/2305.04465 <br>
[3] Tess: https://arxiv.org/pdf/2305.08379 <br>

<br>
<br>

**Supplemented Table 4:** Plug-and-play experiments of Meta-DiffuB on SeqDiffuSeq. The "scheduler" field indicates the dataset our scheduler was originally trained on and trained together with DiffuSeq. The "SeqDiffuSeq" field indicates the dataset the original SeqDiffuSeq model was trained on. If the "SeqDiffuSeq" field is "Null," it means the model used its own noise strategy as described in the original paper. Results where using the scheduler-generated noise during inference outperforms the SeqDiffuSeq's own noise strategy are highlighted in bold.
<table>
  <tr>
    <th>scheduler</th>
    <th>SeqDiffuSeq</th>
    <th>BLEU (↑)</th>
    <th>BERTScore (↑)</th>
    <th>Dist-1 (↑)</th>
  </tr>
  <tr>
    <td>WA</td>
    <td rowspan="3" style="text-align: center">QQP</td>
    <td><strong>0.2627</strong></td>
    <td><strong>0.8481</strong></td>
    <td><strong>0.9814</strong></td>
  </tr>
  <tr>
    <td>QT</td>
    <td><strong>0.2612</strong></td>
    <td><strong>0.8550</strong></td>
    <td><strong>0.9844</strong></td>
  </tr>
  <tr>
    <td>Null</td>
    <td>0.2434</td>
    <td>0.8400</td>
    <td>0.9807</td>
  </tr>
  <tr>
    <td>WA</td>
    <td rowspan="3" style="text-align: center">QT</td>
    <td><strong>0.1834</strong></td>
    <td><strong>0.6226</strong></td>
    <td><strong>0.9369</strong></td>
  </tr>
  <tr>
    <td>QQP</td>
    <td><strong>0.1784</strong></td>
    <td><strong>0.6233</strong></td>
    <td><strong>0.9341</strong></td>
  </tr>
  <tr>
    <td>Null</td>
    <td>0.1746</td>
    <td>0.6174</td>
    <td>0.9248</td>
  </tr>
  <tr>
    <td>QT</td>
    <td rowspan="3" style="text-align: center">WA</td>
    <td><strong>0.3745</strong></td>
    <td><strong>0.8369</strong></td>
    <td><strong>0.9169</strong></td>
  </tr>
  <tr>
    <td>QQP</td>
    <td><strong>0.3827</strong></td>
    <td><strong>0.8257</strong></td>
    <td><strong>0.9158</strong></td>
  </tr>
  <tr>
    <td>Null</td>
    <td>0.3712</td>
    <td>0.8214</td>
    <td>0.9077</td>
  </tr>
</table>
<br>
<br>

**Supplemented Table 5:** More plug-and-play experiments of Meta-DiffuB on Dinoiser. The "scheduler" field indicates the dataset our scheduler was originally trained on and trained together with DiffuSeq. The "Dinoiser" field indicates the dataset the original Dinoiser model was trained on. If the "Dinoiser" field is "Null," it means the model used its own noise strategy as described in the original paper. Results where using the scheduler-generated noise during inference outperforms the Dinoiser's own noise strategy are highlighted in bold.
<table>
  <tr>
    <th>scheduler</th>
    <th>Dinoiser</th>
    <th>BLEU (↑)</th>
    <th>BERTScore (↑)</th>
    <th>Dist-1 (↑)</th>
  </tr>
  <tr>
    <td>WA</td>
    <td rowspan="3" style="text-align: center">QQP</td>
    <td><strong>0.2079</strong></td>
    <td><strong>0.8121</strong></td>
    <td><strong>0.9765</strong></td>
  </tr>
  <tr>
    <td>QT</td>
    <td><strong>0.2092</strong></td>
    <td><strong>0.8207</strong></td>
    <td><strong>0.9660</strong></td>
  </tr>
  <tr>
    <td>Null</td>
    <td>0.1949</td>
    <td>0.8036</td>
    <td>0.9723</td>
  </tr>
  <tr>
    <td>WA</td>
    <td rowspan="3" style="text-align: center">QT</td>
    <td><strong>0.0495</strong></td>
    <td><strong>0.4740</strong></td>
    <td><strong>0.8289</strong></td>
  </tr>
  <tr>
    <td>QQP</td>
    <td><strong>0.0488</strong></td>
    <td><strong>0.4764</strong></td>
    <td><strong>0.8312</strong></td>
  </tr>
  <tr>
    <td>Null</td>
    <td>0.0477</td>
    <td>0.4690</td>
    <td>0.8191</td>
  </tr>
  <tr>
    <td>QT</td>
    <td rowspan="3" style="text-align: center">WA</td>
    <td><strong>0.2409</strong></td>
    <td><strong>0.6885</strong></td>
    <td><strong>0.8368</strong></td>
  </tr>
  <tr>
    <td>QQP</td>
    <td><strong>0.2451</strong></td>
    <td><strong>0.6840</strong></td>
    <td><strong>0.8404</strong></td>
  </tr>
  <tr>
    <td>Null</td>
    <td>0.2388</td>
    <td>0.6787</td>
    <td>0.8421</td>
  </tr>
</table>
