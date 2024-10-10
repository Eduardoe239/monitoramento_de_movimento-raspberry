import os
import time
import subprocess
import cv2
import numpy as np
from datetime import datetime, timedelta
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput

# Inicializa a câmera
picam2 = Picamera2()

# Configurações de vídeo
video_config = picam2.create_video_configuration(
    main={"size": (640, 480), "format": "RGB888"},  # Mantenha 640x480 para melhorar a fluidez
    lores={"size": (640, 480)},
    display="main"
)

# Configura a taxa de quadros
picam2.configure(video_config)
picam2.set_controls({"FrameRate": 30})  # Mantenha 30 fps para boa fluidez

# Diretório para salvar os vídeos
output_dir = "/home/eduardo/Desktop/yolo/videos"
os.makedirs(output_dir, exist_ok=True)  # Cria o diretório se não existir

# Caminhos para os arquivos YOLO
weights_path = "/home/eduardo/Desktop/yolo/yolov3.weights"
config_path = "/home/eduardo/Desktop/yolo/yolov3.cfg"

# Função para converter H264 para MP4 usando FFmpeg
def converter_h264_para_mp4(h264_file, mp4_file):
    comando_ffmpeg = ["ffmpeg", "-i", h264_file, "-c", "copy", mp4_file]
    try:
        subprocess.run(comando_ffmpeg, check=True)
        print(f"Vídeo convertido para MP4: {mp4_file}")
        os.remove(h264_file)
        print(f"Arquivo temporário {h264_file} deletado.")
    except subprocess.CalledProcessError as e:
        print(f"Erro na conversão de {h264_file} para MP4: {e}")

# Função para carregar o modelo YOLO
def carregar_yolo():
    print("Verificando arquivos...")
    print(f"Arquivo de pesos esperado: {weights_path}")
    print(f"Arquivo de configuração esperado: {config_path}")

    if not os.path.exists(weights_path):
        print("Arquivo de pesos não encontrado!")
        return None, None
    if not os.path.exists(config_path):
        print("Arquivo de configuração não encontrado!")
        return None, None

    try:
        net = cv2.dnn.readNet(weights_path, config_path)
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        print("Modelo carregado com sucesso.")
        print("Camadas de saída do YOLO carregadas:", output_layers)  # Log para depuração
        return net, output_layers
    except cv2.error as e:
        print(f"Erro ao carregar o modelo: {e}")
        return None, None

# Função para detectar apenas pessoas
def detectar_pessoas(frame, net, output_layers):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    outputs = net.forward(output_layers)
    print("Outputs recebidos:", len(outputs))  # Verifica a saída

    boxes, confidences, class_ids = [], [], []
    for output in outputs:
        for detection in output:
            scores = detection[5:]  # Assume que os scores começam a partir do índice 5
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if class_id == 0 and confidence > 0.5:  # Classe "person"
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = "Person"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)

# Função para gravar vídeos de 2 horas
def gravar_video():
    print("Iniciando gravação...")
    
    try:
        picam2.start()  # Tenta iniciar a câmera
    except Exception as e:
        print(f"Erro ao iniciar a câmera: {e}")
        return

    intervalo = timedelta(hours=2)
    count = 0

    # Define o horário de término da gravação
    agora = datetime.now()
    fim_gravacao = agora.replace(hour=17, minute=40, second=0, microsecond=0)

    # Carregar modelo YOLO
    net, output_layers = carregar_yolo()
    if net is None or output_layers is None:
        print("Falha ao carregar o modelo YOLO. Encerrando gravação.")
        return

    while agora < fim_gravacao:
        h264_filename = os.path.join(output_dir, f"video_parte_{count}.h264")
        mp4_filename = os.path.join(output_dir, f"video_parte_{count}.mp4")

        encoder = H264Encoder(bitrate=1000000)
        output = FileOutput(h264_filename)

        print(f"Iniciando gravação: {h264_filename} com qualidade otimizada")
        
        # Tenta iniciar a gravação
        try:
            picam2.start_recording(encoder, output)
            print("Gravação iniciada com sucesso.")
        except Exception as e:
            print(f"Erro ao iniciar gravação: {e}")
            picam2.stop()
            return  # Retorna se houver erro

        # Inicializa a janela para exibir o vídeo em tempo real
        cv2.namedWindow("Gravação em Tempo Real", cv2.WINDOW_NORMAL)

        # Grava por 2 horas ou até o horário de término
        tempo_final = min(agora + intervalo, fim_gravacao)
        while datetime.now() < tempo_final:
            # Captura o frame atual da câmera
            frame = picam2.capture_array()  # Captura o frame atual da câmera

            if frame is None:
                print("Frame vazio. Verifique a captura da câmera.")
                continue

            # Chama a função de detecção de objetos (somente pessoas)
            detectar_pessoas(frame, net, output_layers)

            cv2.imshow("Gravação em Tempo Real", frame)  # Mostra o frame na janela
            
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Permite sair pressionando 'q'
                break

        picam2.stop_recording()
        cv2.destroyWindow("Gravação em Tempo Real")  # Fecha a janela

        print(f"Gravação finalizada: {h264_filename}")

        # Verifica se o arquivo foi criado
        if os.path.exists(h264_filename):
            print(f"Arquivo gravado: {h264_filename}")
            # Converte o arquivo H264 para MP4
            converter_h264_para_mp4(h264_filename, mp4_filename)
        else:
            print(f"Arquivo não encontrado após a gravação: {h264_filename}")

        agora = datetime.now()
        count += 1

    picam2.stop()
    print("Gravação concluída.")

# Função principal para controlar o loop de gravação
def loop_gravacao_diaria():
    while True:
        agora = datetime.now()

        # Define os horários de início (5:20) e término (17:40)
        inicio_gravacao = agora.replace(hour=5, minute=20, second=0, microsecond=0)
        fim_gravacao = agora.replace(hour=17, minute=40, second=0, microsecond=0)

        # Verifica se o horário atual está dentro do intervalo de gravação
        if inicio_gravacao <= agora <= fim_gravacao:
            gravar_video()
        else:
            print(f"Fora do horário de gravação, aguardando... {agora.strftime('%Y-%m-%d %H:%M:%S')}")

        # Aguarda 60 segundos antes de verificar novamente
        time.sleep(60)

if __name__ == "__main__":
    loop_gravacao_diaria()
