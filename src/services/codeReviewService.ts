import { ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate } from 'langchain/prompts'
import { LLMChain } from 'langchain/chains'
import { BaseChatModel } from 'langchain/dist/chat_models/base'
import type { ChainValues } from 'langchain/dist/schema'
import { PullRequestFile } from './pullRequestService'
import parseDiff from 'parse-diff'
import { LanguageDetectionService } from './languageDetectionService'
import { exponentialBackoffWithJitter } from '../httpUtils'
import { Effect, Context } from 'effect'
import { NoSuchElementException, UnknownException } from 'effect/Cause'

export interface CodeReviewService {
  codeReviewFor(
    file: PullRequestFile
  ): Effect.Effect<ChainValues, NoSuchElementException | UnknownException, LanguageDetectionService>
  codeReviewForChunks(
    file: PullRequestFile
  ): Effect.Effect<ChainValues, NoSuchElementException | UnknownException, LanguageDetectionService>
}

export const CodeReviewService = Context.GenericTag<CodeReviewService>('CodeReviewService')

export class CodeReviewServiceImpl {
  private llm: BaseChatModel
  private chatPrompt = ChatPromptTemplate.fromPromptMessages([
    SystemMessagePromptTemplate.fromTemplate(
      `Act as an empathetic software engineer that's an expert in designing and developing web application softwares using Java, Springboot framwork and AppDynamics Integration (.yml), and adhering to best practices of software design and architecture.
      You are also an expert in sumarizing the review comments in the form of a predefined report for each and every coding guideline.`
    ),
    HumanMessagePromptTemplate.fromTemplate(`Your task is to review a Pull Request. You will receive a git diff.
    Review it and suggest any improvements in code quality, maintainability, readability, performance, security, etc.
    Identify any potential bugs or security vulnerabilities. After reviewing the code, provide a summary report for each coding guideline if it was followed in the code or not.
    You can refer the following example for a summary report having columns - Code Review checklist, Implemented and Summarization.
    For the Summarization column, add a note on the overall coverage of the respective guideline in the given file.
    Also, provide the report in a tabular format against coding guidelines, at the end of each file in the git diff:
    Example:  Code review checklist	Implemented Summarization
SOLID principles	No
Java Coding Guidelines:
Naming Conventions	No
Indentation and Formatting	No
Comments and Documentation	Yes
Include remaining points here from Java coding guidelines
Springboot Coding Guidelines:
Project Structure:	Yes
Dependency Injection	Yes
RESTful APIs	No
Exception Handling	No
Include remaining points here from Spring boot coding guidelines

Verify that the code adheres to the following design patterns and coding guidelines for Java, Springboot and AppDynamics Integration (.yml), and suggest code improvements accordingly.
-Design Patterns:
1. You have to check if the code follows SOLID principles (Single Responsibility, Open/Closed, Liskov Substitution, Interface Segregation, Dependency Inversion). If not used, suggest how to refactor the code to follow these principles.
-Java Coding Guidelines:
1.Naming Conventions:
a.Class names should be nouns and start with an uppercase letter (e.g., Car, UserService).
b.Method and variable names should be verbs or nouns and start with a lowercase letter (e.g., getUser(), firstName).
c.Constants should be all uppercase with underscores separating words (e.g., MAX_SIZE).
2.Indentation and Formatting:
a.Use 4 spaces for indentation.
b.Use braces even for single-line statements in control structures.
c.Limit lines to 80-120 characters to improve readability.
3.Comments and Documentation:
a.Verify that the Javadoc style comments are used for documenting classes, methods, and fields.
b.Write clear and concise comments to explain complex algorithms or business logic.
4.Exception Handling:
a.Use specific exception types rather than catching Exception.
b.Handle exceptions appropriately, either by logging or throwing further up the call stack.
5.Avoid Magic Numbers and Strings:
a.Verify constants are used instead of hardcoding values.
b.Verify that strings and numbers are defined as constants at the beginning of the class.
6.Immutable Objects:
a.Prefer immutability whenever possible.
b. Verify to make fields final if they should not change after object creation.
7.Use Interfaces and Abstract Classes:
a.Verify the use of interfaces for defining contracts and abstract classes for code reuse.
b.Prefer composition over inheritance.
8.Concurrency:
a.Use thread-safe classes and synchronization mechanisms when dealing with concurrent operations.
b.Utilize Java's concurrent utilities like ExecutorService and ConcurrentHashMap.
-Spring Boot Coding Guidelines:
1.Dependency Injection:
a.Verify the use of constructor injection wherever possible for better testability and immutability.
b.Avoid field injection, prefer setter injection only when required.
2.RESTful APIs:
a. Verify the RESTful principles are followed for designing APIs.
b. Validate the use of appropriate HTTP methods (GET, POST, PUT, DELETE) for CRUD operations.
3.Exception Handling:
a.Verify that @ControllerAdvice is used for global exception handling.
b.Customize error responses using @ExceptionHandler.
4.Security:
a. Verify that the best practices for password hashing and session management are used.
5.Testing:
a.Write unit tests for business logic using frameworks like JUnit and Mockito.
b.Use Spring Boot's testing annotations (@SpringBootTest, @WebMvcTest, etc.) for integration testing.
6.Logging:
a.Use a logging framework like Log4j or Logback.
b.Log meaningful messages with appropriate log levels.
7.Performance:
a.Verify that the database queries are optimized using Spring Data JPA's query methods or custom queries.
b.Cache data using Spring's caching abstraction (@Cacheable, @CacheEvict).
8.Documentation:
a.Document API endpoints using Swagger or Spring REST Docs.
b.Include clear descriptions, request/response examples, and error handling details.
- AppDynamics Integration (.yml) Coding Guidelines:
a.Verify the parameters - appdAgentAppName , appdAgentTierName, appdPlan are defined.
b.Verify that the name for appdAgentTierName follows the naming convention as:  Application name-Application EAI number (for example: FXO-Document-Metadata-Service-3538226 where FXO-Document-Metadata-Service is the application name and 3538226 is the EAI number)

Write your reply and examples in GitHub Markdown format.
The programming language in the git diff is {lang}.

    git diff to review

    {diff}`)
  ])

  private chain: LLMChain<string>

  constructor(llm: BaseChatModel) {
    this.llm = llm
    this.chain = new LLMChain({
      prompt: this.chatPrompt,
      llm: this.llm
    })
  }

  codeReviewFor = (
    file: PullRequestFile
  ): Effect.Effect<ChainValues, NoSuchElementException | UnknownException, LanguageDetectionService> =>
    LanguageDetectionService.pipe(
      Effect.flatMap(languageDetectionService => languageDetectionService.detectLanguage(file.filename)),
      Effect.flatMap(lang =>
        Effect.retry(
          Effect.tryPromise(() => this.chain.call({ lang, diff: file.patch })),
          exponentialBackoffWithJitter(3)
        )
      )
    )

  codeReviewForChunks(
    file: PullRequestFile
  ): Effect.Effect<ChainValues[], NoSuchElementException | UnknownException, LanguageDetectionService> {
    const programmingLanguage = LanguageDetectionService.pipe(
      Effect.flatMap(languageDetectionService => languageDetectionService.detectLanguage(file.filename))
    )
    const fileDiff = Effect.sync(() => parseDiff(file.patch)[0])

    return Effect.all([programmingLanguage, fileDiff]).pipe(
      Effect.flatMap(([lang, fd]) =>
        Effect.all(fd.chunks.map(chunk => Effect.tryPromise(() => this.chain.call({ lang, diff: chunk.content }))))
      )
    )
  }
}
