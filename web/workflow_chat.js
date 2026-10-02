import { app } from "../../scripts/app.js";

const ENDPOINT = "/chatgpt/workflow-chat";
let cleanupPanel = null;

function addStylesheet() {
    if (document.querySelector('link[data-workflow-chat-styles]')) {
        return;
    }
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = new URL("./workflow_chat.css", import.meta.url);
    link.dataset.workflowChatStyles = "";
    document.head.appendChild(link);
}

function currentWorkflow() {
    const workflow = app.graph?.serialize?.();
    if (!workflow || !Array.isArray(workflow.nodes)) {
        throw new Error("Откройте workflow, чтобы задать вопрос о нём.");
    }
    return workflow;
}

function currentErrorContext() {
    const extensionManager = app.extensionManager;
    return {
        node_errors: extensionManager?.lastNodeErrors ?? app.lastNodeErrors ?? null,
        execution_error: extensionManager?.lastExecutionError ?? null,
    };
}

function currentGraphDiagnostics() {
    return {
        unconnected_inputs: (app.graph?._nodes ?? []).flatMap((node) => {
            const requiredInputs = new Set(
                Object.keys(node.constructor?.nodeData?.input?.required ?? {}),
            );
            return (node.inputs ?? [])
                .filter((input) => input.link == null && !input.links?.length)
                .map((input) => ({
                    node_id: node.id,
                    node_type: node.type,
                    node_title: node.title,
                    input_name: input.name,
                    input_type: input.type,
                    required: requiredInputs.has(input.name),
                }));
        }),
    };
}

function renderPanel(container) {
    cleanupPanel?.();
    const abortController = new AbortController();
    const messages = [];

    container.innerHTML = `
        <section class="workflow-chat">
            <header class="workflow-chat__header">
                <div>
                    <strong>Workflow Chat</strong>
                    <span>Workflow и read-only код его нод</span>
                </div>
                <button type="button" class="workflow-chat__clear" title="Очистить чат">
                    Очистить
                </button>
            </header>
            <div class="workflow-chat__messages" aria-live="polite">
                <div class="workflow-chat__empty">
                    Спросите, что делает workflow, где ошибка или как улучшить качество и скорость.
                </div>
            </div>
            <div class="workflow-chat__status" role="status"></div>
            <form class="workflow-chat__form">
                <textarea
                    rows="3"
                    maxlength="6000"
                    placeholder="Например: почему этот workflow медленный?"
                    aria-label="Вопрос о текущем workflow"
                    required
                ></textarea>
                <button type="submit">Отправить</button>
            </form>
        </section>
    `;

    const messageList = container.querySelector(".workflow-chat__messages");
    const status = container.querySelector(".workflow-chat__status");
    const form = container.querySelector(".workflow-chat__form");
    const textarea = form.querySelector("textarea");
    const submitButton = form.querySelector("button");
    const clearButton = container.querySelector(".workflow-chat__clear");

    const addMessage = (role, content) => {
        messageList.querySelector(".workflow-chat__empty")?.remove();
        const bubble = document.createElement("article");
        bubble.className = `workflow-chat__message workflow-chat__message--${role}`;
        if (role === "assistant" && app.extensionManager?.renderMarkdownToHtml) {
            bubble.innerHTML = app.extensionManager.renderMarkdownToHtml(content);
        } else {
            bubble.textContent = content;
        }
        messageList.appendChild(bubble);
        messageList.scrollTop = messageList.scrollHeight;
    };

    const setBusy = (busy) => {
        textarea.disabled = busy;
        submitButton.disabled = busy;
        submitButton.textContent = busy ? "Думаю…" : "Отправить";
        status.textContent = busy ? "Анализирую текущий workflow…" : "";
    };

    form.addEventListener("submit", async (event) => {
        event.preventDefault();
        const question = textarea.value.trim();
        if (!question) {
            return;
        }

        let workflow;
        try {
            workflow = currentWorkflow();
        } catch (error) {
            status.textContent = error.message;
            return;
        }

        messages.push({ role: "user", content: question });
        addMessage("user", question);
        textarea.value = "";
        setBusy(true);

        try {
            const response = await fetch(ENDPOINT, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    messages,
                    workflow,
                    error_context: currentErrorContext(),
                    graph_diagnostics: currentGraphDiagnostics(),
                }),
                signal: abortController.signal,
            });
            const payload = await response.json().catch(() => ({}));
            if (!response.ok) {
                throw new Error(payload.error || `Ошибка сервера: ${response.status}`);
            }
            messages.push({ role: "assistant", content: payload.answer });
            addMessage("assistant", payload.answer);
            status.textContent = "";
        } catch (error) {
            if (error.name !== "AbortError") {
                status.textContent = error.message || "Не удалось получить ответ.";
            }
        } finally {
            setBusy(false);
            textarea.focus();
        }
    });

    textarea.addEventListener("keydown", (event) => {
        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            form.requestSubmit();
        }
    });

    clearButton.addEventListener("click", () => {
        messages.length = 0;
        messageList.innerHTML = `
            <div class="workflow-chat__empty">
                История очищена. Можно задать новый вопрос о текущем workflow.
            </div>
        `;
        status.textContent = "";
        textarea.focus();
    });

    cleanupPanel = () => {
        abortController.abort();
        cleanupPanel = null;
    };
}

app.registerExtension({
    name: "ChatGPT.WorkflowAdvisor",

    async setup() {
        addStylesheet();
        app.extensionManager.registerSidebarTab({
            id: "chatgpt-workflow-advisor",
            icon: "pi pi-comments",
            title: "Workflow Chat",
            tooltip: "Спросить ChatGPT о текущем workflow",
            type: "custom",
            render: (container) => renderPanel(container),
            destroy: () => cleanupPanel?.(),
        });
    },
});
