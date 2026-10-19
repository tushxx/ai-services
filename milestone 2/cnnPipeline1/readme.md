Project/
├── data/
│   ├── train/
│   │   ├── crack/
│   │   ├── inclusion/
│   │   ├── ok/
│   │   └── porosity/
│   ├── test/
│   │   ├── crack/
│   │   ├── inclusion/
│   │   ├── ok/
│   │   └── porosity/
│   └── predict/
├── processed_data/
├── artifacts/
└── output/
└── main.py




predict folder --
    all images that need to be predict or need predictions
        img1.png
        img2.png
        .
        .
        .
        .

train folder --> 
    e.g = classification task
    5 lable, n = 4

    train will have 4 subfolders of those category 
        then that specific subfolder will have different images files

same for test folder 


p = os.path.join('data', 'predict')        --- for path location

p_l = os.listdir(p)                     --- returns a list containing the names of all files and folders inside the directory specified by p

n = ?
d = os.path.join("data", "train")
c_l = os.listdir(d)
n = len(c_l)       ---> 4
 
name of all categories = [i for i in c_l]
[crack, inclusion, ok, porosity]


-------if we want specific folder all images
third = c_l[2]
third_images = os.listdir(os.path.join(d, third))



There are two cases.

Labelled folder
train/
├── crack/
├── inclusion/
├── ok/
└── porosity/
Return:

images, labels
Unlabelled folder
predict/
├── part_000.png
├── part_001.png
└── ...
Return:

images, image_ids