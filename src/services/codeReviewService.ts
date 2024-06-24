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
      "Act as an empathetic software engineer that's an expert in designing and developing web application softwares using Java and Springboot framwork, and adhering to best practices of software design and architecture."
    ),
    HumanMessagePromptTemplate.fromTemplate(`Your task is to review a Pull Request. You will receive a git diff.
    Review it and suggest any improvements in code quality, maintainability, readability, performance, security, etc.
    Identify any potential bugs or security vulnerabilities.
    Check it adheres to the following design patterns and coding guidelines for both Java and Springboot.
   -Design Patterns:
1.Verify that the design patterns like Singleton, Factory, Builder, Strategy and Repository are used where appropriate.
2. If not used, suggest the appropriate design pattern to improve the code.
3. Check if the code follows SOLID principles (Single Responsibility, Open/Closed, Liskov Substitution, Interface Segregation, Dependency Inversion).
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
a.Use Javadoc style comments for documenting classes, methods, and fields.
b.Write clear and concise comments to explain complex algorithms or business logic.
4.Exception Handling:
a.Use specific exception types rather than catching Exception.
b.Handle exceptions appropriately, either by logging or throwing further up the call stack.
5.Avoid Magic Numbers and Strings:
a.Use constants instead of hardcoding values.
b.Define strings and numbers as constants at the beginning of the class.
6.Immutable Objects:
a.Prefer immutability whenever possible.
b.Make fields final if they should not change after object creation.
7.Use Interfaces and Abstract Classes:
a.Use interfaces for defining contracts and abstract classes for code reuse.
b.Prefer composition over inheritance.
8.Concurrency:
a.Use thread-safe classes and synchronization mechanisms when dealing with concurrent operations.
b.Utilize Java's concurrent utilities like ExecutorService and ConcurrentHashMap.
-Spring Boot Coding Guidelines:
1.Project Structure:
a.Organize classes into packages based on their functionality.
b.Follow the standard Maven or Gradle project structure.
2.Dependency Injection:
a.Use constructor injection wherever possible for better testability and immutability.
b.Avoid field injection, prefer setter injection only when required.
3.RESTful APIs:
a.Follow RESTful principles for designing APIs.
b.Use appropriate HTTP methods (GET, POST, PUT, DELETE) for CRUD operations.
4.Exception Handling:
a.Use @ControllerAdvice for global exception handling.
b.Customize error responses using @ExceptionHandler.
5.Security:
a.Use best practices like password hashing and session management.
6.Testing:
a.Write unit tests for business logic using frameworks like JUnit and Mockito.
b.Use Spring Boot's testing annotations (@SpringBootTest, @WebMvcTest, etc.) for integration testing.
7.Logging:
a.Use a logging framework like Log4j or Logback.
b.Log meaningful messages with appropriate log levels.
8.Performance:
a.Optimize database queries using Spring Data JPA's query methods or custom queries.
b.Cache data using Spring's caching abstraction (@Cacheable, @CacheEvict).
9.Documentation:
a.Document API endpoints using Swagger or Spring REST Docs.
b.Include clear descriptions, request/response examples, and error handling details.
10.External Configurations:
a.Externalize configuration using application properties or YAML files.
b.Avoid hardcoding environment-specific values.

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
