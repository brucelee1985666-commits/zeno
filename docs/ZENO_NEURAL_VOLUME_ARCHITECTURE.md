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

## Auditoria inicial dos backends neurais

- tiny-cuda-nn é um candidato concreto para MLPs, codificação hash, treinamento e inferência CUDA; ele não fornece a hierarquia VDB nem a garantia de topologia lossless. O README upstream atual pede CMake 3.21+, enquanto este fork declara CMake 3.15 no projeto raiz. Portanto, não adicionar o upstream atual diretamente como submódulo antes de decidir entre uma versão compatível ou uma atualização deliberada do baseline CMake.
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
