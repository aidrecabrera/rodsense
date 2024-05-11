import inference
model = inference.get_model("cuterat-sense/1")
model.infer(image="img/img.jpg")