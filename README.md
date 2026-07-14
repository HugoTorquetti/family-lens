# FamilyLens

Aplicativo Android de reconhecimento facial **on-device** para identificar membros da família em fotos. Desenvolvido em React Native + TypeScript com módulo nativo Kotlin para inferência de ML.

## O que faz

- Detecta rostos em fotos usando o ML Kit do Google
- Gera embeddings faciais de 512 dimensões via modelo InsightFace ResNet50 (w600k_r50) rodando localmente com ONNX Runtime
- Compara embeddings via distância cosseno para determinar se dois rostos pertencem à mesma pessoa
- 100% on-device — nenhum dado sai do aparelho

## Estado atual

Prova de conceito funcional com interface de comparação 1:1. A lógica de varredura do álbum (1:N) ainda não está implementada.

---

## Pré-requisitos

- [Node.js](https://nodejs.org/) 18+
- [JDK 17](https://adoptium.net/)
- [Android Studio](https://developer.android.com/studio) com SDK Android 35 e NDK instalados
- [Android Emulator](https://developer.android.com/studio/run/emulator) ou dispositivo físico com Android 8+
- Git

> Siga o guia oficial de ambiente antes de continuar: https://reactnative.dev/docs/set-up-your-environment

---

## Instalação

### 1. Clone o repositório

```bash
git clone https://github.com/HugoTorquetti/family-lens.git
cd family-lens
```

### 2. Instale as dependências JS

```bash
npm install
```

### 3. Baixe o modelo InsightFace

O modelo não está no repositório (174 MB). Faça o download e coloque em `android/app/src/main/assets/`:

```
https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/w600k_r50.onnx
```

O arquivo deve ficar em:
```
android/app/src/main/assets/w600k_r50.onnx
```

---

## Rodando o projeto

### Passo 1 — Inicie o emulador

Abra o Android Studio → Device Manager → inicie o emulador (recomendado: Pixel 8, API 35).

> **Importante:** se o emulador travar ao abrir, use a flag `-no-snapshot-load`:
> `emulator -avd <nome_do_avd> -no-snapshot-load`

### Passo 2 — Inicie o Metro Bundler

Em um terminal na raiz do projeto:

```bash
npm start
```

Deixe esse terminal aberto durante todo o desenvolvimento.

### Passo 3 — Compile e instale o APK

Em um **segundo terminal**, na pasta `android/`:

```bash
cd android
.\gradlew installDebug    # Windows
# ou
./gradlew installDebug    # macOS/Linux
```

A primeira build baixa dependências e pode levar alguns minutos. Builds subsequentes são mais rápidas.

> **Windows + OneDrive:** se a build falhar com erros de lock de arquivo, pause o OneDrive antes de buildar e exclua a pasta `android/app/build/` manualmente.

### Passo 4 — Conecte o Metro ao emulador

Se o app abrir com tela branca ou "Unable to load script", rode:

```bash
adb reverse tcp:8081 tcp:8081
```

Depois pressione `R` no emulador para recarregar.

### Passo 5 — Primeira abertura

Na **primeira vez** que o app abre, ele copia o modelo (174 MB) de `assets/` para o armazenamento interno. Isso pode levar **30 a 60 segundos** — a tela fica branca durante esse tempo. É normal. Nas aberturas seguintes o app inicia instantaneamente.

---

## Rodando em dispositivo físico

1. No Android: **Configurações → Opções do desenvolvedor → Depuração USB** (ativar)
2. Conecte o celular via USB e autorize a depuração quando solicitado
3. Confirme que o dispositivo aparece: `adb devices`
4. Execute normalmente: `cd android && .\gradlew installDebug`

> Instalar um APK debug em um dispositivo real é seguro — roda em sandbox isolado sem acesso a dados de outros apps.

---

## Usando o app

1. Toque em **Foto 1** e selecione uma foto da galeria
2. Toque em **Foto 2** e selecione outra foto
3. Toque em **Comparar rostos**
4. O app detecta o rosto principal em cada foto, gera os embeddings e exibe a distância cosseno
   - Distância < 0.72 → **Mesma pessoa**
   - Distância ≥ 0.72 → **Pessoas diferentes**

---

## Arquitetura técnica

```
App.tsx                          # UI em React Native (TypeScript)
  └── FaceRecognition (bridge)   # Módulo nativo via NativeModules

android/app/src/main/java/com/familylens/
  ├── FaceRecognitionModule.kt   # Lógica principal de ML
  ├── FaceRecognitionPackage.kt  # Registro do módulo no React Native
  └── MainApplication.kt

android/app/src/main/assets/
  └── w600k_r50.onnx             # Modelo InsightFace ResNet50 (não versionado)
```

### Pipeline de inferência (`FaceRecognitionModule.kt`)

1. Carrega o bitmap e corrige rotação via EXIF
2. ML Kit detecta todos os rostos na imagem
3. Filtra rostos com baixa probabilidade de olhos abertos (< 0.2) — rejeita bonecas e mascotes
4. Seleciona o rosto com maior `área × max(probOlhoEsq, probOlhoDir)`
5. Aplica 30% de padding no bounding box
6. Alinha o rosto: recorta a região, rotaciona pelo ângulo entre os olhos
7. Redimensiona para 112×112
8. Normaliza pixels: `(pixel - 127.5) / 128.0`, formato NCHW BGR
9. Roda inferência ONNX Runtime → embedding de 512 dimensões
10. Compara dois embeddings via distância cosseno

### Dependências Android principais

```gradle
implementation 'com.google.mlkit:face-detection:16.1.7'
implementation 'com.microsoft.onnxruntime:onnxruntime-android:1.20.0'
implementation 'androidx.exifinterface:exifinterface:1.3.7'
```

---

## Troubleshooting

| Problema | Solução |
|---|---|
| `Unable to load script` | Rode `adb reverse tcp:8081 tcp:8081` e recarregue o app |
| Tela branca na primeira abertura | Aguarde 30–60s — modelo sendo copiado para armazenamento interno |
| Build falha no Windows | Pause o OneDrive e delete `android/app/build/` antes de buildar |
| `adb: no devices found` | Abra o emulador antes de rodar o comando, ou verifique a conexão USB |
| App trava ao abrir | Abra o emulador com `-no-snapshot-load` |
| Rosto errado selecionado | O algoritmo prefere rostos maiores com olhos abertos; fotos com sujeito desencadeado podem selecionar outro rosto |

---

## Debug de rostos detectados

O app salva automaticamente os recortes de todos os rostos detectados e o rosto selecionado em:

```
/sdcard/Android/data/com.familylens/cache/
```

Para inspecionar:

```bash
adb pull /sdcard/Android/data/com.familylens/cache/ ./debug_local/
```

---

## Próximos passos

- [ ] Lógica de varredura 1:N (comparar cada rosto da foto contra N membros cadastrados)
- [ ] Fluxo de enrolamento (cadastrar membros da família com 3–5 fotos de referência)
- [ ] Acesso ao álbum do WhatsApp e varredura em batch
- [ ] UI de revisão antes de deletar fotos
- [ ] Avaliar modelo menor (MobileFaceNet ~4MB) para uso em produção
