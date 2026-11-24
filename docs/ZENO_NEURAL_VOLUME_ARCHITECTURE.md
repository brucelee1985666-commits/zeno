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

A amostragem de minibatches agora usa um `torch.Generator` CUDA local por domínio, inicializado com a seed derivada da chave espacial. Classificador e regressor usam seeds distintas e estáveis. Isso remove a dependência da sequência global de sorteios entre domínios. A reprodutibilidade bit a bit ainda depende da versão do PyTorch/CUDA, GPU e kernels determinísticos disponíveis; portanto, não é prometida entre hardwares diferentes.

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
