import { config } from 'dotenv'
import * as core from '@actions/core'
import * as github from '@actions/github'
import type { PullRequestEvent } from '@octokit/webhooks-definitions/schema'

import { ChatOpenAI } from 'langchain/chat_models/openai'
import { BaseChatModel } from 'langchain/dist/chat_models/base'
import { CodeReviewService, CodeReviewServiceImpl } from './services/codeReviewService'
import { PullRequestService, PullRequestServiceImpl, octokitTag } from './services/pullRequestService'
import { LanguageDetectionService } from './services/languageDetectionService'
import { Effect, Layer, Match, pipe, Exit } from 'effect'

// Imports for Claude
// import { ChatAnthropic } from 'langchain/chat_models/anthropic' // Updated import
// import { foldLeftChunksEffect } from 'effect/Sink'

config()

export const run = async (): Promise<void> => {
  const openAIApiKey = core.getInput('openai_api_key')
  const githubToken = core.getInput('github_token')
  const modelName = core.getInput('model_name')
  const temperature = parseInt(core.getInput('model_temperature'))
  // const azureOpenAIApiKey = core.getInput('azure_openai_api_key')
  // const azureOpenAIApiInstanceName = core.getInput('azure_openai_api_instance_name')
  // const azureOpenAIApiDeploymentName = core.getInput('azure_openai_api_deployment_name')
  // const azureOpenAIApiVersion = core.getInput('azure_openai_api_version')
  //const anthropic_api_key = core.getInput('anthropic_api_key')
  //const claudeModelName = 'claude-3-7-sonnet-20250219'
  const context = github.context
  const { owner, repo } = context.repo

  const model: BaseChatModel = new ChatOpenAI({
    temperature,
    openAIApiKey,
    modelName
  })

  // // Claude model initialization
  // const model: BaseChatModel = new ChatAnthropic({ modelName: claudeModelName,
  // temperature: 0.7,
  // anthropicApiKey: anthropic_api_key
  // })
  // console.log('Model initialized' + model)

  const MainLive = initializeServices(model, githubToken)

  const program = Match.value(context.eventName).pipe(
    Match.when('pull_request', () => {
      const excludeFilePatterns = pipe(
        Effect.sync(() => github.context.payload as PullRequestEvent),
        Effect.tap(pullRequestPayload =>
          Effect.sync(() => {
            core.info(
              `repoName: ${repo} pull_number: ${context.payload.number} owner: ${owner} sha: ${pullRequestPayload.pull_request.head.sha}`
            )
          })
        ),
        Effect.map(() =>
          core
            .getInput('exclude_files')
            .split(',')
            .map(_ => _.trim())
        )
      )

      const a = excludeFilePatterns.pipe(
        Effect.flatMap(filePattens =>
          PullRequestService.pipe(
            Effect.flatMap(pullRequestService =>
              pullRequestService.getFilesForReview(owner, repo, context.payload.number, filePattens)
            ),
            Effect.flatMap(files => Effect.sync(() => files.filter(file => file.patch !== undefined))),
            Effect.flatMap(files =>
              Effect.forEach(files, file =>
                CodeReviewService.pipe(
                  Effect.flatMap(codeReviewService => codeReviewService.codeReviewFor(file)),
                  Effect.flatMap(res =>
                    PullRequestService.pipe(
                      Effect.flatMap(pullRequestService =>
                        pullRequestService.createReviewComment({
                          repo,
                          owner,
                          pull_number: context.payload.number,
                          commit_id: context.payload.pull_request?.head.sha,
                          path: file.filename,
                          body: res.text,
                          subject_type: 'file'
                        })
                      )
                    )
                  )
                )
              )
            )
          )
        )
      )

      return a
    }),

    Match.orElse(eventName =>
      Effect.sync(() => {
        core.setFailed(`This action only works on pull_request events. Got: ${eventName}`)
      })
    )
  )

  const runnable = Effect.provide(program, MainLive)
  const result = await Effect.runPromiseExit(runnable)

  if (Exit.isFailure(result)) {
    core.setFailed(result.cause.toString())
  }
}

const initializeServices = (model: BaseChatModel, githubToken: string) => {
  const CodeReviewServiceLive = Layer.effect(
    CodeReviewService,
    Effect.map(LanguageDetectionService, _ => CodeReviewService.of(new CodeReviewServiceImpl(model)))
  )

  const octokitLive = Layer.succeed(octokitTag, github.getOctokit(githubToken))

  const PullRequestServiceLive = Layer.effect(
    PullRequestService,
    Effect.map(octokitTag, _ => PullRequestService.of(new PullRequestServiceImpl()))
  )

  const mainLive = CodeReviewServiceLive.pipe(
    Layer.merge(PullRequestServiceLive),
    Layer.merge(LanguageDetectionService.Live),
    Layer.merge(octokitLive),
    Layer.provide(LanguageDetectionService.Live),
    Layer.provide(octokitLive)
  )

  return mainLive
}

run()
