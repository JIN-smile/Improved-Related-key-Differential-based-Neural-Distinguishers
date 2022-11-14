# Improved (Related-key) Differential-based Neural Distinguishers for SIMON and SIMECK Block Ciphers<br> 
**`Title`**：Improved (Related-key) Differential-based Neural Distinguishers for SIMON and SIMECK Block Ciphers<br> 
**`Author`**: Jinyu Lu, Guoqiang Liu, Bing Sun, Chao Li and Li Liu<br> 
**`Abstract`**：In CRYPTO 2019, Gohr made a pioneering attempt and successfully applied deep learning to the differential cryptanalysis against NSA block cipher Speck32/64, achieving higher accuracy than the pure differential distinguishers. By its very nature, mining effective features in data plays a crucial role in data-driven deep learning. In this paper, in addition to considering the integrity of the
information from the training data of the ciphertext pair, domain knowledge about the structure of differential cryptanalysis is also considered into the training process of deep learning to improve the performance. Meanwhile, taking the performance of the differential-neural distinguisher of Simon32/64 as an entry point, we investigate the impact of input difference on the performance of the hybrid distinguishers to choose the proper input difference. Eventually, we improve the accuracy of the neural distinguishers of Simon32/64, Simon64/128, Simeck32/64, and Simeck64/128. We also obtain related-key differential-based neural distinguishers on round-reduced versions of Simon32/64, Simon64/128, Simeck32/64, and Simeck64/128 for the first time.<br><br>
**`Tested configuration`**<br>
python == 3.9.13<br>
tensorflow == 2.5.0<br>
h5py == 3.1.0<br>
numpy == 1.19.5<br>
pandas == 1.4.4<br>
