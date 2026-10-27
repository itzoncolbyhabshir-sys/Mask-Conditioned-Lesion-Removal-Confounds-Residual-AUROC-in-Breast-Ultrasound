# BIBM C4 dilation-scale audit

C4 uses the saved dilated lesion mask in the original image/mask space before encoder-specific resizing.

| Dataset | Images | Width median [IQR] | Height median [IQR] | Area increase median [IQR] | Kernel/short edge median [IQR] | Radius/lesion diameter median [IQR] |
| --- | ---: | --- | --- | --- | --- | --- |
| BUS-BRA | 1875 | 286.00 [276.00, 323.00] | 383.00 [350.00, 435.00] | 9.91% [7.08%, 13.60%] | 1.75% [1.55%, 1.81%] | 2.03% [1.47%, 2.83%] |
| BUS-UCLM | 260 | 856.00 [856.00, 856.00] | 606.00 [606.00, 606.00] | 5.15% [3.77%, 7.49%] | 0.83% [0.83%, 0.83%] | 1.09% [0.78%, 1.63%] |
