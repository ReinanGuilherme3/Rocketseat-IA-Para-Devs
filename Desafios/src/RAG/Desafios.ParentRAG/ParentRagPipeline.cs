using Desafios.RAG.Shared;
using Microsoft.Extensions.AI;
using Pgvector;

namespace Desafios.ParentRAG;

/// <summary>
/// Parent RAG: a busca vetorial roda nos chunks pequenos (mais fáceis de "casar"
/// semanticamente com a pergunta), mas o contexto enviado ao LLM é o chunk pai
/// inteiro — mais texto ao redor do trecho relevante, o que costuma ajudar o modelo
/// a responder perguntas que dependem de contexto mais amplo do que uma frase isolada.
/// </summary>
public class ParentRagPipeline(
    ParentChildChunkStore store,
    IEmbeddingGenerator<string, Embedding<float>> embeddingGenerator,
    IChatClient chatClient)
{
    public async Task<string> AnswerAsync(string question, int topParents = 4)
    {
        // 1) Pergunta -> embedding.
        var questionEmbedding = await embeddingGenerator.GenerateAsync([question]);
        var queryVector = new Vector(questionEmbedding[0].Vector.ToArray());

        // 2) Busca pelos filhos mais próximos, mas recupera o texto do pai de cada um.
        var retrievedParents = await store.FindNearestParentsAsync(queryVector, topParents);
        var context = string.Join("\n\n---\n\n", retrievedParents);

        // 3) LLM responde usando o contexto (mais amplo) dos chunks pai.
        var prompt = AnswerPromptBuilder.Build(context, question);
        var response = await chatClient.GetResponseAsync(prompt);

        return response.Text;
    }
}
