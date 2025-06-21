# Notas e Observações

Não se esquecer de marcar as tarefas como concluidas.

sempre usar o mesmo modelo para se criar uma nova tarefa.

Identificar e agrupar por versões.

Antes de partir para a próxima versão fazer uma tag no git indicando a versão atual

Documente todo o código de forma detalhada e didática, classes, funções e algoritmos que se distacam.

# TODO

## Versão 1.0

- [X] Ajustar o layout de cores dos botões mantendo padronizado
- [ ] Criar arquivo TUTORIAL.md com informações didáticas sobre o modelo escolhido, cnn. Explicar porque os demais modelos não funciona, seja minucioso e didático nas explicações.
- [ ] Adicionar chave de comando para que já inicialize a análise de arquivos de teste, sendo que neste caso será obrigatório que o arquivo de modelo já exista e seja informado em conjunto
- [ ] Adicionar chave de comando para que já inicialize a classificação de arquivos de teste, sendo que neste caso será obrigatório que o arquivo de modelo já exista e seja informado em conjunto
- [ ] Adicionar chave de comando para que já inicialize a treinamento, mas neste caso se for informado o nome do aquivo de modelo salva sobre ele e se não existir cria.
- [x] Adicionar contagem de segundos por arquivo e média
- [x] Contar bytes por arquivo e média
- [x] Contar arquivos processados
- [ ] obter mais estátisticas
- [x] Adicionar um botão para tocar os ruidos, abrirá um modal com os botões agrupados para selecionar os ruidos da pasta train e da pasta test, e por sua vez agrupados pelas pastas que representam as classes, cria box com títulos para cada agrupamento.
- [x] Refatorar modal de ruídos para exibir abas 'Treino' e 'Teste' com sub-abas por classe (Normal, Intermediário, Falha) e listas roláveis de arquivos
- [x] Adicionar tracker de tempo de reprodução e botão Parar/Continuar no modal de ruídos

## Versão 2.0

- [ ] Ajustar o script de criação h5 para stm32, para criar o projeto conforme o processador escolhido, família ou placa de protipação, criar toda a estrutura inclusive o arquivo ioc adequado.
- [ ] Mover todos os parametros staticos para um arquivo de configuração config.py na pasta simulador.
- [ ] Mover todos os parametros de layout para um arquivo layout.py na pasta simulador.
- [ ] Adicionar um splash screen que mostra o nome do projeto, os autores, a versão, esta informações estão no arquivo version.py na pasta simulador.
- [ ] Ajustar o parametro versão para 2.0
- [ ] Adicionar novas redes neurais
- [ ] Adicionar as teclas de saida do sistema de forma graciosa(CTRL+Q), e se for na linha de comando (CTRL+C)


## Versão 3.0

- [ ] Adicionar um menu que permite mudar os diretórios de obtenção de arquivos de treino e teste.
- [ ] Adicionar um menu que permite escolher onde gravar os modelos
- [ ] Adicionar um menu que permite escolher o modelo a ser usado e em qual diretório está
- [ ] Adicionar um menu que permite escolher onde será gravado o relatório

## Versão 4.0
- O branch de trabalho no git da versão 4.0 é o branch servidor_RESTfull.
- De tarefa em tarefa concluida deve fazer o git add e git commit com a mensagem da tarefa concluida e descrita.

- [X] Permitir conexão via restfull, obedecendo o padrão da API da OpenAI, para receber arquivos de áudio e retornar a classificação.
- [X] Gerar um cliente para o servidor RESTfull que seja compatível com o uso como funções do Open Web UI, na pasta RESTfull.
- [X] o arquivo rest_client.py deve estar na pasta RESTfull. Remova o arquivo rest_client.py da pasta simulador.
- [X] Adicionar os seguintes metadados que devem ser retornados pelo servidor RESTfull: Formatos de áudio suportado WAV, MP3, OGG, comprimento do arquivo, formato do áudio, amostragem utilizada
- [X] Gerar o swarmm.json para o servidor RESTfull, na pasta RESTfull.
- [X] Documentar em RESTfull.md como usar o servidor RESTfull
- [X] adicionar a chave de comando --rest para iniciar o servidor RESTfull como serviço do linux usando o systemd, crie um arquivo de serviço de exemplo na pasta RESTfull.
- [X] Quando usar a chave --rest pode-se também usar a chave --host e/ou --port que informa o ip a ser ligado o servidor RESTfull e a porta que escutará.
- [X] Ajustar o cliente para que ele se expelhe na classe "função do tipo action do open webui" de exemplo "action.py" que está na pasta RESTfull. Ajustes os comentários para que o autor seja Carlos Delfino, e o email seja consultoria@carlosdelfino.eti.br, versão 0.1.0, Titulo Analisador de Ruidos de Magnetostricção, Open Web UI requirido na versão 0.3.9
- [X] Criar um script para instalar o serviço RESTfull

## Versão 4.1
Sempre tratar os arquivos de audio como brutos, e não urls ou paths.

- [X] Corrigir a requisição de envio do audio, deve ser o áudio codificado, ele deve vir em formato bruto como wave, mp3 ou ogg.

