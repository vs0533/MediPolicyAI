"use client";

import { useEffect, useRef, useState } from "react";
import type { KeyboardEvent, RefObject } from "react";
import Image from "next/image";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";
import {
  Bot,
  BookOpen,
  CheckCircle2,
  ChevronDown,
  FileText,
  FileSearch,
  Loader2,
  Plus,
  Send,
  ShieldCheck,
  Square,
  Trash2,
  User,
} from "lucide-react";
import { useGlobal } from "@/context/GlobalContext";
import { processLatexContent } from "@/lib/latex";
import {
  APP_NAME,
  APP_SUBTITLE,
  APP_VERSION,
  POLICY_KNOWLEDGE_LABEL,
  POLICY_KNOWLEDGE_PATH,
} from "@/lib/public-service";
import { apiUrl } from "@/lib/api";

type PopularQuestion = {
  question: string;
  count: number;
  last_asked_at?: string;
};

type QuestionSuggestions = {
  preset_questions: string[];
  popular_questions: PopularQuestion[];
};

const FALLBACK_PRESET_QUESTIONS = [
  "医保报销范围怎么判断？",
  "异地就医备案怎么办理？",
  "门诊慢特病政策有哪些？",
];

export default function HomePage() {
  const {
    chatState,
    setChatState,
    sendChatMessage,
    stopChatMessage,
    newChatSession,
    clearChatHistory,
  } = useGlobal();
  const [inputMessage, setInputMessage] = useState("");
  const [questionSuggestions, setQuestionSuggestions] = useState<QuestionSuggestions>({
    preset_questions: FALLBACK_PRESET_QUESTIONS,
    popular_questions: [],
  });
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const loadQuestionSuggestions = async () => {
    try {
      const response = await fetch(apiUrl("/api/v1/chat/question-suggestions?limit=3&popular_limit=15"));
      const payload = await response.json();
      const data = payload?.data;
      if (!payload?.success || !data) return;
      setQuestionSuggestions({
        preset_questions:
          Array.isArray(data.preset_questions) && data.preset_questions.length > 0
            ? data.preset_questions
            : FALLBACK_PRESET_QUESTIONS,
        popular_questions: Array.isArray(data.popular_questions) ? data.popular_questions : [],
      });
    } catch {
      setQuestionSuggestions((prev) => ({
        ...prev,
        preset_questions: prev.preset_questions.length > 0 ? prev.preset_questions : FALLBACK_PRESET_QUESTIONS,
      }));
    }
  };

  useEffect(() => {
    setChatState((prev) => ({
      ...prev,
      enableRag: true,
      enableWebSearch: false,
      selectedKb: POLICY_KNOWLEDGE_PATH,
      searchMode: "FAST",
    }));
    loadQuestionSuggestions();
  }, [setChatState]);

  useEffect(() => {
    const el = messagesContainerRef.current;
    if (!el) return;
    const { scrollTop, scrollHeight, clientHeight } = el;
    const nearBottom = scrollHeight - scrollTop - clientHeight < 200;
    if (!nearBottom && !chatState.isLoading) return;
    requestAnimationFrame(() => {
      if (messagesContainerRef.current) {
        messagesContainerRef.current.scrollTop = messagesContainerRef.current.scrollHeight;
      }
    });
  }, [chatState.messages, chatState.isLoading]);

  const handleSend = () => {
    const message = inputMessage.trim();
    if (!message || chatState.isLoading) return;
    sendChatMessage(message);
    setInputMessage("");
    window.setTimeout(loadQuestionSuggestions, 1000);
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.nativeEvent.isComposing || event.key === "Process") return;
    if (event.key === "Enter") {
      event.preventDefault();
      handleSend();
    }
  };

  const hasMessages = chatState.messages.length > 0;

  return (
    <div className="h-screen flex flex-col bg-slate-50 dark:bg-slate-900">
      <header className="border-b border-slate-200 dark:border-slate-700 bg-white/90 dark:bg-slate-800/90 backdrop-blur-sm">
        <div className="max-w-5xl mx-auto px-6 py-4 flex items-center justify-between gap-4">
          <div className="flex items-center gap-3 min-w-0">
            <Image
              src="/logo.svg"
              alt={`${APP_NAME} Logo`}
              width={40}
              height={40}
              className="object-contain shrink-0"
              priority
            />
            <div className="min-w-0">
              <h1 className="text-lg font-semibold text-slate-900 dark:text-slate-100 truncate">
                {APP_NAME}
              </h1>
              <p className="text-sm text-slate-500 dark:text-slate-400 truncate">
                {APP_SUBTITLE}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2 shrink-0">
            <span className="rounded-full border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900 px-2 py-1 text-xs font-medium text-slate-500 dark:text-slate-400">
              {APP_VERSION}
            </span>
            <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-full bg-teal-50 dark:bg-teal-900/30 border border-teal-100 dark:border-teal-700 text-teal-700 dark:text-teal-300 text-sm">
              <ShieldCheck className="w-4 h-4" />
              <span>{POLICY_KNOWLEDGE_LABEL}</span>
            </div>
          </div>
        </div>
      </header>

      <main ref={messagesContainerRef} className="flex-1 overflow-y-auto">
        {!hasMessages ? (
          <div className="min-h-full flex items-center justify-center px-6 py-10">
            <div className="w-full max-w-3xl">
              <div className="text-center mb-8">
                <div className="flex items-center justify-center mb-5">
                  <Image
                    src="/logo.svg"
                    alt={`${APP_NAME} Logo`}
                    width={72}
                    height={72}
                    className="object-contain"
                    priority
                  />
                </div>
                <h2 className="text-3xl font-semibold text-slate-900 dark:text-slate-100 mb-3">
                  医保政策问答
                </h2>
                <p className="text-base text-slate-500 dark:text-slate-400">
                  面向公众提供医保政策咨询、条款解释和办理指引。
                </p>
              </div>
              <QuestionInput
                value={inputMessage}
                disabled={chatState.isLoading}
                inputRef={inputRef}
                onChange={setInputMessage}
                onKeyDown={handleKeyDown}
                onSend={handleSend}
                onStop={stopChatMessage}
                isLoading={chatState.isLoading}
              />
              <div className="mt-5 grid grid-cols-1 sm:grid-cols-3 gap-3">
                {questionSuggestions.preset_questions.slice(0, 3).map((text) => (
                  <button
                    key={text}
                    onClick={() => setInputMessage(text)}
                    className="text-left px-4 py-3 rounded-lg bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-sm text-slate-600 dark:text-slate-300 hover:border-teal-300 dark:hover:border-teal-600 hover:text-teal-700 dark:hover:text-teal-300 transition-colors"
                  >
                    {text}
                  </button>
                ))}
              </div>
              {questionSuggestions.popular_questions.length > 0 && (
                <section className="mt-6">
                  <div className="mb-2 flex items-center justify-between gap-3">
                    <h3 className="text-sm font-medium text-slate-600 dark:text-slate-300">
                      大家常问
                    </h3>
                    <span className="text-xs text-slate-400 dark:text-slate-500">
                      按提问频率排序
                    </span>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {questionSuggestions.popular_questions.slice(0, 15).map((item) => (
                      <button
                        key={item.question}
                        onClick={() => setInputMessage(item.question)}
                        className="inline-flex max-w-full items-center gap-1.5 rounded-full border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 px-3 py-1.5 text-xs text-slate-600 dark:text-slate-300 hover:border-teal-300 dark:hover:border-teal-600 hover:text-teal-700 dark:hover:text-teal-300 transition-colors"
                        title={`已被提问 ${item.count} 次`}
                      >
                        <span className="truncate">{item.question}</span>
                        <span className="shrink-0 text-slate-400 dark:text-slate-500">
                          {item.count}
                        </span>
                      </button>
                    ))}
                  </div>
                </section>
              )}
            </div>
          </div>
        ) : (
          <div className="max-w-5xl mx-auto px-6 py-6 space-y-6">
            {chatState.messages.map((msg, index) => (
              <div key={index} className="flex gap-4">
                <div
                  className={`w-9 h-9 rounded-full flex items-center justify-center shrink-0 ${
                    msg.role === "user"
                      ? "bg-slate-200 dark:bg-slate-700"
                      : "bg-teal-600 text-white"
                  }`}
                >
                  {msg.role === "user" ? (
                    <User className="w-4 h-4 text-slate-500 dark:text-slate-300" />
                  ) : (
                    <Bot className="w-4 h-4" />
                  )}
                </div>
                <div className="flex-1 min-w-0">
                  <div
                    className={`px-5 py-4 rounded-2xl border ${
                      msg.role === "user"
                        ? "bg-slate-100 dark:bg-slate-800 border-slate-200 dark:border-slate-700 text-slate-800 dark:text-slate-200"
                        : "bg-white dark:bg-slate-800 border-slate-200 dark:border-slate-700 text-slate-800 dark:text-slate-200 shadow-sm"
                    }`}
                  >
                    <div className="prose prose-slate dark:prose-invert prose-sm max-w-none prose-table:text-xs prose-th:bg-slate-50 dark:prose-th:bg-slate-900 prose-td:align-top">
                      <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>
                        {processLatexContent(msg.content)}
                      </ReactMarkdown>
                    </div>
                    {msg.isStreaming && (
                      <div className="flex items-center gap-2 mt-3 text-teal-600 dark:text-teal-300 text-sm">
                        <Loader2 className="w-4 h-4 animate-spin" />
                        <span>正在查询医保政策知识库...</span>
                      </div>
                    )}
                  </div>
                  {msg.role === "assistant" && (msg.searchLogs?.length || msg.isStreaming) && (
                    <SearchProcessPanel logs={msg.searchLogs || []} isActive={Boolean(msg.isStreaming)} />
                  )}
                  {msg.role === "assistant" &&
                    msg.sources?.references &&
                    msg.sources.references.length > 0 && (
                      <details
                        open
                        className="mt-2 rounded-lg border border-teal-100 dark:border-teal-800 bg-white dark:bg-slate-800"
                      >
                        <summary className="flex items-center gap-2 px-3 py-2 cursor-pointer text-xs font-medium text-teal-700 dark:text-teal-300">
                          <BookOpen className="w-3.5 h-3.5 text-teal-500" />
                          参考依据：已命中 {msg.sources.references.length} 个政策文件
                        </summary>
                        <div className="divide-y divide-slate-100 dark:divide-slate-700">
                          {msg.sources.references.map((ref, refIndex) => (
                            <div key={refIndex} className="px-3 py-2.5 space-y-1.5">
                              <div className="flex items-start gap-1.5 text-xs font-medium text-slate-700 dark:text-slate-300">
                                <FileText className="w-3.5 h-3.5 text-teal-500 shrink-0" />
                                <span className="break-words">{formatReferenceTitle(ref.file, refIndex)}</span>
                              </div>
                              {ref.summary && (
                                <p className="text-xs text-slate-600 dark:text-slate-400 leading-relaxed">
                                  {cleanReferenceText(ref.summary, 220)}
                                </p>
                              )}
                              {ref.snippets?.slice(0, 2).map((snippet: string, snippetIndex: number) => (
                                <p
                                  key={snippetIndex}
                                  className="text-xs bg-slate-50 dark:bg-slate-900 text-slate-600 dark:text-slate-400 px-2 py-1.5 rounded leading-relaxed"
                                >
                                  {cleanReferenceText(snippet, 260)}
                                </p>
                              ))}
                            </div>
                          ))}
                        </div>
                      </details>
                    )}
                </div>
              </div>
            ))}
          </div>
        )}
      </main>

      {hasMessages && (
        <footer className="border-t border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 px-6 py-4">
          <div className="max-w-5xl mx-auto space-y-3">
            <div className="flex items-center justify-between gap-3">
              <div className="flex items-center gap-2 text-sm text-slate-500 dark:text-slate-400">
                <ShieldCheck className="w-4 h-4 text-teal-500" />
                <span>{POLICY_KNOWLEDGE_LABEL}</span>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={clearChatHistory}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs text-slate-500 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-700"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                  清空
                </button>
                <button
                  onClick={newChatSession}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs text-slate-500 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-700"
                >
                  <Plus className="w-3.5 h-3.5" />
                  新对话
                </button>
              </div>
            </div>
            <QuestionInput
              value={inputMessage}
              disabled={chatState.isLoading}
              inputRef={inputRef}
              onChange={setInputMessage}
              onKeyDown={handleKeyDown}
              onSend={handleSend}
              onStop={stopChatMessage}
              isLoading={chatState.isLoading}
            />
          </div>
        </footer>
      )}
    </div>
  );
}

function SearchProcessPanel({
  logs,
  isActive,
}: {
  logs: Array<{
    level: string;
    message: string;
    timestamp: string;
    is_streaming?: boolean;
    task_id?: string;
    flush?: boolean;
  }>;
  isActive: boolean;
}) {
  const visibleLogs = buildPublicSearchLogs(logs, isActive);

  if (visibleLogs.length === 0 && !isActive) return null;

  return (
    <details
      open={isActive}
      className="mt-2 rounded-lg border border-teal-100 dark:border-teal-800 bg-teal-50/60 dark:bg-teal-950/20"
    >
      <summary className="flex items-center justify-between gap-3 px-3 py-2 cursor-pointer text-xs font-medium text-teal-700 dark:text-teal-300">
        <span className="flex items-center gap-2">
          {isActive ? (
            <Loader2 className="w-3.5 h-3.5 animate-spin" />
          ) : (
            <CheckCircle2 className="w-3.5 h-3.5" />
          )}
          检索过程
        </span>
        <ChevronDown className="w-3.5 h-3.5" />
      </summary>
      <div className="px-3 pb-3 space-y-2">
        {visibleLogs.map((log, index) => (
          <div key={`${log}-${index}`} className="flex items-start gap-2 text-xs text-slate-600 dark:text-slate-300">
            <FileSearch className="w-3.5 h-3.5 mt-0.5 text-teal-500 shrink-0" />
            <span className="leading-relaxed">{log}</span>
          </div>
        ))}
      </div>
    </details>
  );
}

function buildPublicSearchLogs(
  logs: Array<{ level: string; message: string }>,
  isActive: boolean,
) {
  const items: string[] = [];
  const seen = new Set<string>();

  const push = (value: string) => {
    const text = value.trim();
    if (!text || seen.has(text)) return;
    seen.add(text);
    items.push(text);
  };

  if (isActive || logs.length > 0) {
    push("正在分析问题并检索医保政策知识库");
  }

  for (const log of logs) {
    const raw = sanitizeSearchLog(log.message);
    if (!raw) continue;

    if (raw.includes("[FAST:Step1] Primary")) {
      push("已提取检索关键词");
      continue;
    }
    if (raw.includes("[FAST:Step2] Best file")) {
      const file = extractDisplayFileName(raw);
      push(file ? `已找到候选文件：${file}` : "已找到候选政策文件");
      push("正在校验候选依据是否足够回答");
      continue;
    }
    if (raw.includes("[FAST:Step3] Evidence")) {
      const chars = raw.match(/Evidence:\s*(\d+)\s*chars/i)?.[1];
      push(chars ? `已抽取政策依据片段（约 ${chars} 字符）` : "已抽取政策依据片段");
      continue;
    }
    if (raw.includes("[FAST:Step4]") || raw.includes("Generating")) {
      push("正在依据检索结果生成回答");
      continue;
    }
    if (
      raw.includes("Evidence acceptance: False") ||
      raw.includes("Evidence rejected after retry") ||
      raw.includes("Candidate files were found")
    ) {
      push("候选文件中找到相关片段，但不足以支撑该问题的明确答案");
      continue;
    }
    if (raw.includes("Search complete") || raw.includes("Knowledge base search completed")) {
      push("知识库检索完成");
      continue;
    }
    if (log.level === "error" || raw.includes("failed")) {
      push("检索过程中出现异常，已尝试降级处理");
    }
  }

  if (!isActive && logs.length > 0) {
    push("回答已生成");
  }

  return items.slice(-8);
}

function sanitizeSearchLog(message: string) {
  return (message || "")
    .replace(/<think>[\s\S]*?<\/think>/gi, "")
    .replace(/\[role=assistant\][\s\S]*/gi, "")
    .replace(/\/Users\/[^\s,'")\]]+/g, "政策知识库")
    .replace(/\s+/g, " ")
    .trim();
}

function extractDisplayFileName(message: string) {
  const match = message.match(/Best file \([^)]+\):\s*(.*?)\s*\(/);
  return match?.[1]?.trim();
}

function formatReferenceTitle(file: string, index: number) {
  const name = file.split("/").pop()?.trim();
  return name || `政策依据 ${index + 1}`;
}

function cleanReferenceText(value: string, maxLength: number) {
  const text = (value || "")
    .replace(/<think>[\s\S]*?<\/think>/gi, "")
    .replace(/\[role=assistant\][\s\S]*/gi, "")
    .replace(/```(?:json|markdown)?\s*([\s\S]*?)```/gi, "$1")
    .replace(/\s+/g, " ")
    .trim();

  if (text.length <= maxLength) return text;
  return `${text.slice(0, maxLength).replace(/[，,；;。.\s]+$/, "")}...`;
}

function QuestionInput({
  value,
  disabled,
  inputRef,
  isLoading,
  onChange,
  onKeyDown,
  onSend,
  onStop,
}: {
  value: string;
  disabled: boolean;
  isLoading: boolean;
  onChange: (value: string) => void;
  inputRef: RefObject<HTMLInputElement | null>;
  onKeyDown: (event: KeyboardEvent<HTMLInputElement>) => void;
  onSend: () => void;
  onStop: () => void;
}) {
  return (
    <div className="relative">
      <input
        ref={inputRef}
        type="text"
        className="w-full px-5 py-4 pr-14 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-xl focus:outline-none focus:ring-2 focus:ring-teal-500/20 focus:border-teal-500 transition-all placeholder:text-slate-400 dark:placeholder:text-slate-500 text-slate-700 dark:text-slate-200 shadow-sm"
        placeholder="请输入医保政策相关问题..."
        value={value}
        onChange={(event) => onChange(event.target.value)}
        onKeyDown={onKeyDown}
        disabled={disabled}
      />
      {isLoading ? (
        <button
          onClick={onStop}
          className="absolute right-2 top-2 bottom-2 aspect-square bg-red-500 text-white rounded-lg flex items-center justify-center hover:bg-red-600 transition-all"
          title="停止生成"
        >
          <Square className="w-4 h-4 fill-current" />
        </button>
      ) : (
        <button
          onClick={onSend}
          disabled={!value.trim()}
          className="absolute right-2 top-2 bottom-2 aspect-square bg-teal-600 text-white rounded-lg flex items-center justify-center hover:bg-teal-700 disabled:opacity-50 disabled:hover:bg-teal-600 transition-all"
          title="发送"
        >
          <Send className="w-5 h-5" />
        </button>
      )}
    </div>
  );
}
