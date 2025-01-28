import gradio as gr
import yaml
from src.load_model import model, device

config = yaml.safe_load(open('config.yaml', 'r'))
threshold = config['predct']['threshold']

def predict(text: str):
    prediction = model(text).item()
    label = "Negative" if prediction >= threshold else "Positive"
    return label, float(prediction)  

examples = [
    ["Спасибо за подробный разбор, это действительно полезно!"],
    ["Интересный подход, я бы добавил ещё пару примеров для наглядности."],
    ["Никогда не задумывался об этом с такой точки зрения. Подумаю над вашей идеей."],
    ["папа вроде нормальным был а сынок говнюком вырос."],
    ["говно на палке блять чё красивого в этой картинке"],
    ["идиоты! что попало придумывают лишь бы лайки ставили"]
]

interface = gr.Interface(
    fn=predict,
    title="Text Classification",
    description=f"using device: {device}",
    inputs=gr.Textbox(label="Текст для классификации"),
    outputs=[
        gr.Textbox(label="Класс", interactive=False),
        gr.Slider(minimum=0, maximum=1, label="Оценка модели", interactive=False)
    ],
    live=True,
    examples=examples
)

if __name__ == "__main__":
    interface.launch()