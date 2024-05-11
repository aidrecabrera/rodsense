from ultralytics import YOLO, checks, hub
checks()

hub.login('c8a3b61f0f83d93bb5b7871c6df1b30ab6d82f3310')

model = YOLO('https://hub.ultralytics.com/models/NB7kRw4U5qeo8gPFlp83')
results = model.train()