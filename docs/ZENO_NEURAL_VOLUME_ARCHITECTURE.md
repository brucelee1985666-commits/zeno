# Zeno Neural Volume — arquitetura e plano de implementação

Status: RFC inicial; implementação em andamento. Esta especificação não afirma paridade com NVIDIA NeuralVDB.

## Objetivo

Criar uma representação compacta de volumes esparsos do Zeno, compatível com a pipeline OpenVDB, que separe:

1. topologia espacial;
2. valores (densidade, SDF e vetores);
3. metadados e transformações;
4. estado temporal entre frames.

O alvo funcional é reproduzir os princípios demonstrados em NeuralVDB: High-resolution Sparse Volume Representation using Hierarchical Neural Networks (Kim, Lee e Museth, ACM TOG 2024): hierarquia VDB explícita nas regiões superiores, classificadores neurais para ocupação/topologia nos níveis inferiores, regressores para valores e warm-start entre frames. A topologia precisa ser reconstruível sem falsos negativos/falsos positivos; qualquer predição aproximada deve ser acompanhada de uma estrutura explícita de correção.

## Regras de arquitetura

- OpenVDB continua sendo a representação editável e a fonte de verdade da simulação.
- NanoVDB é um snapshot de leitura/consulta, não substitui a topologia mutável de OpenVDB.
- A representação neural é um codec/cache derivado, nunca a única cópia editável.
- O runtime deve suportar consultas espaciais seletivas; não exigir decodificar o volume inteiro para cada amostra.
- A codificação é offline/assíncrona; a renderização não deve treinar redes.
- O modo de compressão e a tolerância de erro são metadados explícitos por grid.
- Cada grid guarda tipo, transform, classe, bounding box, voxel size, versão do codec e checksums.
- Não adicionar frameworks de ML ou submódulos antes de confirmar licença, ABI, custo de build e integração CUDA.

## Estado encontrado no fork

Branch de trabalho: zeno-master-clean-copy, inicialmente idêntica à master.

- projects/zenvdb já contém os nós OpenVDB e a classe VDBGridWrapper.
- projects/zenvdb/CMakeLists.txt adiciona o submódulo OpenVDB diretamente.
- O submódulo fixado é zenustech/openvdb no commit 015847945a09ff41db27fbd8602ae14b31e27293, cuja árvore contém NanoVDB e os exemplos oficiais OpenVDB→NanoVDB e CUDA.
- O OpenVDB fixado declara versão 9.0.1. Não se deve presumir compatibilidade binária com versões mais novas.
- projects/CUDA integra ZPC e CUDA, mas depende de ZENO_WITH_zenvdb e ZENO_WITH_ZenoFX.
- Não foi encontrada integração NanoVDB nos nós atuais de projects/zenvdb.

## Pipeline alvo

    OpenVDB editável
        |
        +--> NanoVDB snapshot (baseline de leitura/consulta)
        |
        +--> Neural Volume Encoder (offline)
                |
                +-- níveis superiores VDB explícitos
                +-- classificador de ocupação/topologia
                +-- regressor(es) de valores
                +-- correções explícitas para garantir topologia exata
                +-- warm-start / estado temporal opcional
                |
                +--> arquivo neural versionado
                         |
                         +--> consultas seletivas / decode por bloco
                         +--> reconstrução OpenVDB quando solicitada
                         +--> amostragem para CUDA/OptiX

## Estratégia de implementação por gates

### Gate 0 — integração de referência (em andamento)

Adicionar nós Zeno para converter FloatGrid OpenVDB em NanoVDB e gravar .nvdb, e para importar .nvdb de volta como FloatGrid OpenVDB, usando os headers e conversores já presentes no submódulo fixado. Isso estabelece um caminho de ida e volta sem importar uma dependência nova. A primeira entrega usa codec de arquivo NONE e precisão integral; quantização/ZIP/BLOSC serão opções separadas após confirmar a configuração real do build.

### Gate 1 — contrato do codec neural

Definir um formato versionado que armazene:
- cabeçalho/magic/version/endianness;
- tipo de grid e metadados espaciais;
- hierarquia explícita e índices dos blocos;
- pesos e descritores dos modelos por bloco;
- dados exatos de correção da topologia;
- tolerância/escala/offset dos valores;
- estado de warm-start e referência ao frame anterior;
- checksums e tamanhos para validação.

### Gate 2 — codificação neural estática

Começar com FloatGrid de densidade e SDF. Preservar níveis superiores da árvore; treinar classificadores e regressores locais nos níveis inferiores. A ocupação deve ser lossless por correções explícitas verificáveis. Os valores podem ser lossy com limites de erro definidos pelo usuário. Comparar pelo menos contra OpenVDB comprimido e NanoVDB quantizado.

### Gate 3 — inferência seletiva em GPU

Implementar consulta por bloco sem materializar o volume inteiro. Avaliar tiny-cuda-nn apenas como backend de MLP/hash encoding; não presumir que ele forneça a hierarquia ou o codec. O layout do modelo deve ser independente do backend para permitir kernels próprios se forem necessários.

### Gate 4 — animação

Adicionar warm-start sequencial de pesos, política de novos blocos e detecção de mudanças de topologia. Nunca reutilizar pesos sem validar bounds, transform e identidade do bloco.

### Gate 1A — dataset de treino ZNVDS v1 (implementado; ainda sem validação C++)

Foi adicionado o nó `ExportNeuralVolumeDataset` em `projects/zenvdb/ExportNeuralVolumeDataset.cpp`. Ele aceita FloatGrid e escreve um dataset de treino binário `.znvds` sem materializar o volume inteiro em memória.

O ZNVDS v1 contém:
- magic, versão e flags explícitas;
- nome e classe do grid (para distinguir, por exemplo, fog volume de level set), background e voxel size;
- matriz affine index-to-world 4×4 em double, com rejeição de transforms não lineares;
- tiles ativos internos preservados como registros explícitos;
- uma máscara de ocupação exata de 512 bits por leaf 8³;
- valores float32 dos voxels ativos, indexados por offset local de 16 bits.

A separação é intencional: a máscara é o alvo do classificador de ocupação; os valores ativos são o alvo do regressor. Os tiles superiores permanecem explícitos nesta primeira versão. O dataset é um intercâmbio de treinamento, não é ainda um arquivo NeuralVDB comprimido e não armazena pesos treinados.

Também foram adicionados `tools/neural_volume/inspect_znvds.py` e testes unitários em `test_inspect_znvds.py`, além de um workflow Python restrito à branch de trabalho para validar o parser. O smoke check local do layout sintético passou; isso não valida a compilação nem a saída real do nó C++.

Limitações conhecidas do ZNVDS v1:
- FloatGrid somente;
- somente transforms lineares/affine;
- valores inativos de voxels não são exportados como targets de regressão;
- o parser valida estrutura e contagens, mas ainda não compara uma reconstrução contra o grid original.

### Gate 2A — treinador neural experimental (implementado; GPU ainda não executada)

Foi adicionado `tools/neural_volume/train_znvds.py`, usando a extensão Python do tiny-cuda-nn:
- um HashGrid + CutlassMLP para classificar ocupação por domínio;
- outro HashGrid + CutlassMLP para regredir valores ativos;
- tiles internos e lista de leaf blocks permanecem explícitos;
- diferenças de classificação são guardadas como correções esparsas, para reconstrução exata da máscara quando o mesmo backend/configuração de inferência for usado;
- valores cuja previsão excede a tolerância definida são guardados como correções exatas;
- domínios pequenos que não justificam um regressor guardam os valores explicitamente;
- métricas incluem accuracy, IoU, precision/recall, contagem de correções e RMSE pré-correção.

A saída `.znvm` agora é um contêiner binário little-endian com parâmetros tiny-cuda-nn e registros de correção, acompanhado de um inspetor Python. Ainda não existe decoder C++ do Zeno para esse contêiner. A correção de topologia depende de inferência numericamente compatível com a usada durante a codificação; isso precisa ser validado antes de prometer topologia lossless entre plataformas. O contêiner registra os hiperparâmetros do modelo, mas a integração C++ ainda precisa recriar a mesma configuração tiny-cuda-nn e carregar seus parâmetros.

Dependências do treinamento: Python, NumPy, PyTorch com CUDA e tiny-cuda-nn instalado conforme as instruções upstream. A extensão tiny-cuda-nn exige GPU NVIDIA e não oferece fallback CPU. O treinamento real não foi executado nesta etapa porque não há uma GPU de execução disponível nesta sessão.

### Reprodutibilidade do treinamento (ajuste incremental)

A amostragem de minibatches agora usa um `torch.Generator` CUDA local por domínio, inicializado com a seed derivada da chave espacial. Classificador e regressor usam seeds distintas e estáveis. A seed global de treinamento e a indicação desse modo de amostragem também são gravadas no JSON de metadados do ZNVM. Isso remove a dependência da sequência global de sorteios entre domínios. A reprodutibilidade bit a bit ainda depende da versão do PyTorch/CUDA, GPU e kernels determinísticos disponíveis; portanto, não é prometida entre hardwares diferentes.

O leitor C++ também rejeita offsets de correção duplicados, pesos/parâmetros não finitos, métricas não finitas e valores de tiles/correções inválidos. Fixtures negativas cobrem duplicatas e NaN antes de conectar o runtime CUDA.

### Gate 2B — leitor C++ do contêiner ZNVM (implementado e compilado em CI)

Foi adicionado `projects/zenvdb/NeuralVolumeModel.h/.cpp`:
- lê o cabeçalho versionado, hash da origem, metadados espaciais e configuração do modelo;
- carrega parâmetros do classificador/regressor, tiles explícitos e correções por leaf;
- valida limites de tamanho, contagens, flags, transform, offsets e bytes excedentes;
- não depende de tiny-cuda-nn para ler/validar o contêiner.

O teste C++ `tools/neural_volume/znvm_reader_test.cpp` compila e executa em CI, cobrindo arquivo completo, arquivo parcial e rejeição de bytes excedentes. Isso valida o leitor binário com fixtures sintéticas; ainda não valida arquivos produzidos por uma sessão real de treinamento nem a inferência CUDA.

## Auditoria inicial dos backends neurais

- tiny-cuda-nn é usado agora como backend experimental do treinador Python para HashGrid + CutlassMLP; ele não fornece a hierarquia VDB nem a garantia de topologia lossless. O upstream atual pede CMake 3.18+ para seu build C++ e a extensão Python depende de PyTorch com CUDA. Por isso, não foi adicionado como submódulo nem como dependência obrigatória do runtime Zeno.
- ƒVDB é uma infraestrutura esparsa de GPU integrada ao ecossistema PyTorch. É candidato para prototipagem/treinamento offline, mas não deve entrar no runtime C++ do Zeno como dependência pesada nesta fase.
- ZPC já está presente no fork e fornece infraestrutura CUDA/memória. Reutilizar seus recursos onde couber, sem criar um segundo alocador ou runtime paralelo.
- A decisão de backend deve ser tomada após fixar uma versão, conferir licença, CUDA/CMake mínimos e validar o caminho de exportação/importação de pesos. A arquitetura do arquivo neural permanece independente do backend.

## Métricas obrigatórias

- razão de compressão em disco e em memória;
- tempo de codificação por frame;
- tempo de decodificação/reconstrução;
- latência de consulta aleatória e amostragem por bloco;
- erro RMSE/MAE para densidade e SDF;
- IoU/recall/precision da ocupação e contagem de correções;
- flicker temporal em sequências;
- VRAM e tráfego CPU↔GPU.

Não declarar vitória por razão de compressão isolada: NeuralVDB prioriza footprint e aceita possível perda de velocidade de acesso.

## Fora do escopo inicial

- substituir OpenVDB na simulação;
- introduzir VQVDB/dense2vdb como dependências;
- implementar NeuralVDB inteiro numa única alteração;
- treinar redes no caminho de renderização;
- afirmar desempenho superior sem benchmark reproduzível.

## Referências técnicas

- Kim, Lee, Museth, NeuralVDB, ACM TOG 43(2), 2024: https://doi.org/10.1145/3641817
- Código do Zeno: https://github.com/brucelee1985666-commits/zeno
- OpenVDB/NanoVDB submódulo usado pelo fork: https://github.com/zenustech/openvdb/tree/015847945a09ff41db27fbd8602ae14b31e27293/nanovdb/nanovdb



### Validação confiável sem GPU NVIDIA

O caminho de validação sem GPU foi ampliado sem fingir que tiny-cuda-nn roda em
CPU:

- `inspect_znvds.py` usa apenas a biblioteca padrão do Python para verificar
  finitude dos valores, alinhamento/duplicação de leaves, ocupação versus
  amostras escalares, atribuição de domínio inclusive para coordenadas
  negativas, amostras nos planos de fronteira e um SHA-256 dos registros
  ativos em ordem de arquivo.
- `nanovdb_cpu_reference_smoke.cpp` usa o `GridBuilder` CPU oficial do
  NanoVDB, converte com `createNanoGrid` e verifica valores de voxels ativos
  e de um tile ativo em um NanoVDB FloatGrid real.
- Esses testes exercitam parsing, topologia e valores de referência no CPU.
  Eles não executam HashGrid/MLP, não validam os pesos tiny-cuda-nn e não
  substituem inferência CUDA.

O upstream tiny-cuda-nn exige GPU NVIDIA e CUDA para treinamento/inferência.
Por isso ele continua opcional. A CI separada usa o toolkit CUDA para
compilação sem GPU; compilação não é execução. Não existe alegação de
treinamento, inferência ou benchmark neural realizado sem hardware compatível.

## Estado consolidado após a integração do decoder CUDA

A implementação agora contém um caminho funcional em código para recriar redes
tiny-cuda-nn a partir da configuração e dos parâmetros ZNVM, fazer inferência
por lotes, decodificar leaves seletivamente e produzir um NanoVDB FloatGrid
residente na GPU com topologia nativa de leaves e tiles internos. O decoder mantém uma cache LRU limitada a dois domínios para evitar uploads
repetidos ao atravessar fronteiras próximas, mantendo o uso de VRAM limitado. O caminho de construção NanoVDB
ainda consulta a ocupação das leaves na CPU para montar a topologia, enquanto
os valores escalares são mantidos em buffers GPU.

### Correções de fidelidade e continuidade

- O caminho de decodificação com montagem no host aplica correções de ocupação
  antes de escolher as amostras do regressor. Isso evita que um falso negativo corrigido para ativo
  receba incorretamente o valor de background.
- O treinador conserva os valores ativos de uma casca de uma leaf (8 voxels)
  nas seis faces de cada domínio. Assim, as faces de domínios vizinhos usam
  valores de origem exatos. O interior continua usando o regressor e correções
  exatas apenas quando excede a tolerância.
- Esta política é um fallback exato de fronteira, não blending entre redes.
  Ela reduz o risco de seams entre domínios sem adicionar uma segunda rede ou
  uma nova dependência.

### Limites que continuam explícitos

- A hierarquia espacial é a hierarquia VDB explícita preservada (tiles e
  leaves) mais o particionamento em domínios neurais. Ainda não há uma
  hierarquia neural multinível que aprenda a escolher resoluções/modelos em
  diferentes níveis como no desenho completo do NeuralVDB.
- O caminho NanoVDB reconstrói a topologia no host e copia máscaras de
  ocupação leaf a leaf. É um primeiro bridge híbrido, não uma atualização
  totalmente GPU-resident nem um caminho otimizado para streaming de frames.
- A cache de redes mantém até dois domínios por decoder; chamadas concorrentes
  precisam ser serializadas por instância.
- O formato e o runtime continuam restritos a FloatGrid e ao backend/
  configuração tiny-cuda-nn gravados no checkpoint.
- A CI compila e executa testes CPU do formato e compila separadamente o
  bridge CUDA sem exigir GPU. Isso não substitui inferência real em hardware,
  round-trip contra OpenVDB, nem medições de compressão, qualidade, latência e
  VRAM.
- Warm-start temporal entre frames e integração como nó ZenoCompute/render
  ainda não foram concluídos.

Portanto, o resultado é um protótipo híbrido alinhado aos princípios centrais
do NeuralVDB (topologia explícita, valores neurais locais e correções exatas),
não uma reprodução integral ou uma alegação de paridade com o SDK NeuralVDB.
