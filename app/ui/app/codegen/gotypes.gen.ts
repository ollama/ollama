/* Do not change, this code is generated from Golang structs */


export class ChatInfo {
    id: string;
    title: string;
    userExcerpt: string;
    createdAt: Date;
    updatedAt: Date;

    constructor(source: any = {}) {
        if ('string' === typeof source) source = JSON.parse(source);
        this.id = source["id"];
        this.title = source["title"];
        this.userExcerpt = source["userExcerpt"];
        this.createdAt = new Date(source["createdAt"]);
        this.updatedAt = new Date(source["updatedAt"]);
    }
}
export class ChatsResponse {
    chatInfos: ChatInfo[];

    constructor(source: any = {}) {
        if ('string' === typeof source) source = JSON.parse(source);
        this.chatInfos = this.convertValues(source["chatInfos"], ChatInfo);
    }

	convertValues(a: any, classs: any, asMap: boolean = false): any {
	    if (!a) {
	        return a;
	    }
	    if (Array.isArray(a)) {
	        return (a as any[]).map(elem => this.convertValues(elem, classs));
	    } else if ("object" === typeof a) {
	        if (asMap) {
	            for (const key of Object.keys(a)) {
	                a[key] = new classs(a[key]);
	            }
	            return a;
	        }
	        return new classs(a);
	    }
	    return a;
	}
}
export class Time {


    constructor(source: any = {}) {
        if ('string' === typeof source) source = JSON.parse(source);

    }
}
export class ToolFunction {
    name: string;
    arguments: string;
    result?: any;

    constructor(source: any = {}) {
        if ('string' === typeof source) source = JSON.parse(source);
        this.name = source["name"];
        this.arguments = source["arguments"];
        this.result = source["result"];
    }
}
export class ToolCall {
    type: string;
    function: ToolFunction;

    constructor(source: any = {}) {
        if ('string' === typeof source) source = JSON.parse(source);
        this.type = source["type"];
        this.function = this.convertValues(source["function"], ToolFunction);
    }

	convertValues(a: any, classs: any, asMap: boolean = false): any {
	    if (!a) {
	        return a;
	    }
	    if (Array.isArray(a)) {
	        return (a as any[]).map(elem => this.convertValues(elem, classs));
	    } else if ("object" === typeof a) {
	        if (asMap) {
	            for (const key of Object.keys(a)) {
	                a[key] = new classs(a[key]);
	            }
	            return a;
	        }
	        return new classs(a);
	    }
	    return a;
	}
}
export class File {
    filename: string;
    data: number[];

    constructor(source: any = {}) {
        if ('string' === typeof source) source = JSON.parse(source);
        this.filename = source["filename"];
        this.data = source["data"];
    }
}
export class Message {
    role: string;
    content: string;
    thinking: string;
    stream: boolean;
    model?: string;
    attachments?: File[];
    tool_calls?: ToolCall[];
    tool_call?: ToolCall;
    tool_name?: string;
    tool_result?: number[];
    created_at: Time;
    updated_at: Time;
    thinkingTimeStart?: Date | undefined;
    thinkingTimeEnd?: Date | undefined;

    constructor(source: any = {}) {
        if ('string' === typeof source) source = JSON.parse(source);
        this.role = source["role"];
        this.content = source["content"];
        this.thinking = source["thinking"];
        this.stream = source["stream"];
        this.model = source["model"];
        this.attachments = this.convertValues(source["attachments"], File);
        this.tool_calls = this.convertValues(source["tool_calls"], ToolCall);
        this.tool_call = this.convertValues(source["tool_call"], ToolCall);
        this.tool_name = source["tool_name"];
        this.tool_result = source["tool_result"];
        this.created_at = this.convertValues(source["created_at"], Time);
        this.updated_at = this.convertValues(source["updated_at"], Time);
        this.thinkingTimeStart = source["thinkingTimeStart"] && new Date(source["thinkingTimeStart"]);
        this.thinkingTimeEnd = source["thinkingTimeEnd"] && new Date(source["thinkingTimeEnd"]);
    }

	convertValues(a: any, classs: any, asMap: boolean = false): any {
	    if (!a) {
	        return a;
	    }
	    if (Array.isArray(a)) {
	        return (a as any[]).map(elem => this.convertValues(elem, classs));
	    } else if ("object" === typeof a) {
	        if (asMap) {
	            for (const key of Object.keys(a)) {
	                a[key] = new classs(a[key]);
	            }
	            return a;
	        }
	        return new classs(a);
	    }
	    return a;
	}
}
export class Chat {
    id: string;
    messages: Message[];
    title: string;
    created_at: Time;
    browser_state?: BrowserStateData;

    constructor(source: any = {}) {
        if ('string' === typeof source) source = JSON.parse(source);
        this.id = source["id"];
        this.messages = this.convertValues(source["messages"], Message);
        this.title = source["title"];
        this.created_at = this.convertValues(source["created_at"], Time);
        this.browser_state = source["browser_state"];
    }

	convertValues(a: any, classs: any, asMap: boolean = false): any {
	    if (!a) {
	        return a;
	    }
	    if (Array.isArray(a)) {
	        return (a as any[]).map(elem => this.convertValues(elem, classs));
	    } else if ("object" === typeof a) {
	        if (asMap) {
	            for (const key of Object.keys(a)) {
	                a[key] = new classs(a[key]);
	            }
	            return a;
	        }
	        return new classs(a);
	    }
	    return a;
	}
}
export class ChatResponse {
    chat: Chat;

    constructor(source: any = {}) {
        if ('string' === typeof source) source = JSON.parse(source);
        this.chat = this.convertValues(source["chat"], Chat);
    }

	convertValues(a: any, classs: any, asMap: boolean = false): any {
	    if (!a) {
	        return a;
	    }
	    if (Array.isArray(a)) {
	        return (a as any[]).map(elem => this.convertValues(elem, classs));
	    } else if ("object" === typeof a) {
	        if (asMap) {
	            for (const key of Object.keys(a)) {
	                a[key] = new classs(a[key]);
	            }
	            return a;
	        }
	        return new classs(a);
	    }
	    return a;
	}
}
export class Model {
    model: string;
    digest?: string;
    modified_at?: Time;

    constructor(source: any = {}) {
        if ('string' === typeof source) source = JSON.parse(source);
        this.model = source["model"];
        this.digest = source["digest"];
        this.modified_at = this.convertValues(source["modified_at"], Time);
    }

	convertValues(a: any, classs: any, asMap: boolean = false): any {
	    if (!a) {
	        return a;
	    }
	    if (Array.isArray(a)) {
	        return (a as any[]).map(elem => this.convertValues(elem, classs));
	    } else if ("object" === typeof a) {
	        if (asMap) {
	            for (const key of Object.keys(a)) {
	                a[key] = new classs(a[key]);
	            }
	            return a;
	        }
	        return new classs(a);
	    }
	    return a;
	}
}
export class ModelsResponse {
    models: Model[];

    constructor(source: any = {}) {
        if ('string' === typeof source) source = JSON.parse(source);
        this.models = this.convertValues(source["models"], Model);
    }

	convertValues(a: any, classs: any, asMap: boolean = false): any {
	    if (!a) {
	        return a;
	    }
	    if (Array.isArray(a)) {
	        return (a as any[]).map(elem => this.convertValues(elem, classs));
	    } else if ("object" === typeof a) {
	        if (asMap) {
	            for (const key of Object.keys(a)) {
	                a[key] = new classs(a[key]);
	            }
	            return a;
	        }
	        return new classs(a);
	    }
	    return a;
	}
}
export class InferenceCompute {
    library: string;
    variant: string;
    compute: string;
    driver: string;
    name: string;
    vram: string;

    constructor(source: any = {}) {
        if ('string' === typeof source) source = JSON.parse(source);
        this.library = source["library"];
        this.variant = source["variant"];
        this.compute = source["compute"];
        this.driver = source["driver"];
        this.name = source["name"];
        this.vram = source["vram"];
    }
}
export class InferenceComputeResponse {
    inferenceComputes: InferenceCompute[];

    constructor(source: any = {}) {
        if ('string' === typeof source) source = JSON.parse(source);
        this.inferenceComputes = this.convertValues(source["inferenceComputes"], InferenceCompute);
    }

	convertValues(a: any, classs: any, asMap: boolean = false): any {
	    if (!a) {
	        return a;
	    }
	    if (Array.isArray(a)) {
	        return (a as any[]).map(elem => this.convertValues(elem, classs));
	    } else if ("object" === typeof a) {
	        if (asMap) {
	            for (const key of Object.keys(a)) {
	                a[key] = new classs(a[key]);
	            }
	            return a;
	        }
	        return new classs(a);
	    }
	    return a;
	}
}
export class ModelCapabilitiesResponse {
    capabilities: string[];

    constructor(source: any = {}) {
        if ('string' === typeof source) source = JSON.parse(source);
        this.capabilities = source["capabilities"];
    }
}
export class ChatEvent {
    eventName: "chat" | "thinking" | "assistant_with_tools" | "tool_call" | "tool" | "tool_result" | "done" | "chat_created";
    content?: string;
    thinking?: string;
    thinkingTimeStart?: Date | undefined;
    thinkingTimeEnd?: Date | undefined;
    toolCalls?: ToolCall[];
    toolCall?: ToolCall;
    toolName?: string;
    toolResult?: boolean;
    toolResultData?: any;
    chatId?: string;
    toolState?: any;

    constructor(source: any = {}) {
        if ('string' === typeof source) source = JSON.parse(source);
        this.eventName = source["eventName"];
        this.content = source["content"];
        this.thinking = source["thinking"];
        this.thinkingTimeStart = source["thinkingTimeStart"] && new Date(source["thinkingTimeStart"]);
        this.thinkingTimeEnd = source["thinkingTimeEnd"] && new Date(source["thinkingTimeEnd"]);
        this.toolCalls = this.convertValues(source["toolCalls"], ToolCall);
        this.toolCall = this.convertValues(source["toolCall"], ToolCall);
        this.toolName = source["toolName"];
        this.toolResult = source["toolResult"];
        this.toolResultData = source["toolResultData"];
        this.chatId = source["chatId"];
        this.toolState = source["toolState"];
    }

	convertValues(a: any, classs: any, asMap: boolean = false): any {
	    if (!a) {
	        return a;
	    }
	    if (Array.isArray(a)) {
	        return (a as any[]).map(elem => this.convertValues(elem, classs));
	    } else if ("object" === typeof a) {
	        if (asMap) {
	            for (const key of Object.keys(a)) {
	                a[key] = new classs(a[key]);
	            }
	            return a;
	        }
	        return new classs(a);
	    }
	    return a;
	}
}
export class DownloadEvent {
    eventName: "download";
    total: number;
    completed: number;
    done: boolean;

    constructor(source: any = {}) {
        if ('string' === typeof source) source = JSON.parse(source);
        this.eventName = source["eventName"];
        this.total = source["total"];
        this.completed = source["completed"];
        this.done = source["done"];
    }
}
export class ErrorEvent {
    eventName: "error";
    error: string;
    code?: string;
    details?: string;

    constructor(source: any = {}) {
        if ('string' === typeof source) source = JSON.parse(source);
        this.eventName = source["eventName"];
        this.error = source["error"];
        this.code = source["code"];
        this.details = source["details"];
    }
}
export class Settings {
    Expose: boolean;
    Browser: boolean;
    Survey: boolean;
    Models: string;
    Agent: boolean;
    Tools: boolean;
    WorkingDir: string;
    ContextLength: number;
    AirplaneMode: boolean;
    TurboEnabled: boolean;
    WebSearchEnabled: boolean;
    ThinkEnabled: boolean;
    ThinkLevel: string;
    SelectedModel: string;
    SidebarOpen: boolean;

    constructor(source: any = {}) {
        if ('string' === typeof source) source = JSON.parse(source);
        this.Expose = source["Expose"];
        this.Browser = source["Browser"];
        this.Survey = source["Survey"];
        this.Models = source["Models"];
        this.Agent = source["Agent"];
        this.Tools = source["Tools"];
        this.WorkingDir = source["WorkingDir"];
        this.ContextLength = source["ContextLength"];
        this.AirplaneMode = source["AirplaneMode"];
        this.TurboEnabled = source["TurboEnabled"];
        this.WebSearchEnabled = source["WebSearchEnabled"];
        this.ThinkEnabled = source["ThinkEnabled"];
        this.ThinkLevel = source["ThinkLevel"];
        this.SelectedModel = source["SelectedModel"];
        this.SidebarOpen = source["SidebarOpen"];
    }
}
export class SettingsResponse {
    settings: Settings;

    constructor(source: any = {}) {
        if ('string' === typeof source) source = JSON.parse(source);
        this.settings = this.convertValues(source["settings"], Settings);
    }

	convertValues(a: any, classs: any, asMap: boolean = false): any {
	    if (!a) {
	        return a;
	    }
	    if (Array.isArray(a)) {
	        return (a as any[]).map(elem => this.convertValues(elem, classs));
	    } else if ("object" === typeof a) {
	        if (asMap) {
	            for (const key of Object.keys(a)) {
	                a[key] = new classs(a[key]);
	            }
	            return a;
	        }
	        return new classs(a);
	    }
	    return a;
	}
}
export class HealthResponse {
    healthy: boolean;

    constructor(source: any = {}) {
        if ('string' === typeof source) source = JSON.parse(source);
        this.healthy = source["healthy"];
    }
}
export class User {
    id: string;
    email: string;
    name: string;
    bio?: string;
    avatarurl?: string;
    firstname?: string;
    lastname?: string;
    plan?: string;

    constructor(source: any = {}) {
        if ('string' === typeof source) source = JSON.parse(source);
        this.id = source["id"];
        this.email = source["email"];
        this.name = source["name"];
        this.bio = source["bio"];
        this.avatarurl = source["avatarurl"];
        this.firstname = source["firstname"];
        this.lastname = source["lastname"];
        this.plan = source["plan"];
    }
}
export class Attachment {
    filename: string;
    data?: string;

    constructor(source: any = {}) {
        if ('string' === typeof source) source = JSON.parse(source);
        this.filename = source["filename"];
        this.data = source["data"];
    }
}
export class ChatRequest {
    model: string;
    prompt: string;
    index?: number;
    attachments?: Attachment[];
    web_search?: boolean;
    file_tools?: boolean;
    forceUpdate?: boolean;
    think?: any;

    constructor(source: any = {}) {
        if ('string' === typeof source) source = JSON.parse(source);
        this.model = source["model"];
        this.prompt = source["prompt"];
        this.index = source["index"];
        this.attachments = this.convertValues(source["attachments"], Attachment);
        this.web_search = source["web_search"];
        this.file_tools = source["file_tools"];
        this.forceUpdate = source["forceUpdate"];
        this.think = source["think"];
    }

	convertValues(a: any, classs: any, asMap: boolean = false): any {
	    if (!a) {
	        return a;
	    }
	    if (Array.isArray(a)) {
	        return (a as any[]).map(elem => this.convertValues(elem, classs));
	    } else if ("object" === typeof a) {
	        if (asMap) {
	            for (const key of Object.keys(a)) {
	                a[key] = new classs(a[key]);
	            }
	            return a;
	        }
	        return new classs(a);
	    }
	    return a;
	}
}
export class Error {
    error: string;

    constructor(source: any = {}) {
        if ('string' === typeof source) source = JSON.parse(source);
        this.error = source["error"];
    }
}
export class ModelUpstreamResponse {
    digest?: string;
    pushTime: number;
    error?: string;

    constructor(source: any = {}) {
        if ('string' === typeof source) source = JSON.parse(source);
        this.digest = source["digest"];
        this.pushTime = source["pushTime"];
        this.error = source["error"];
    }
}
export class Page {
    url: string;
    title: string;
    text: string;
    lines: string[];
    links?: Record<number, string>;
    fetched_at: Time;

    constructor(source: any = {}) {
        if ('string' === typeof source) source = JSON.parse(source);
        this.url = source["url"];
        this.title = source["title"];
        this.text = source["text"];
        this.lines = source["lines"];
        this.links = source["links"];
        this.fetched_at = this.convertValues(source["fetched_at"], Time);
    }

	convertValues(a: any, classs: any, asMap: boolean = false): any {
	    if (!a) {
	        return a;
	    }
	    if (Array.isArray(a)) {
	        return (a as any[]).map(elem => this.convertValues(elem, classs));
	    } else if ("object" === typeof a) {
	        if (asMap) {
	            for (const key of Object.keys(a)) {
	                a[key] = new classs(a[key]);
	            }
	            return a;
	        }
	        return new classs(a);
	    }
	    return a;
	}
}
export class BrowserStateData {
    page_stack: string[];
    view_tokens: number;
    url_to_page: {[key: string]: Page};

    constructor(source: any = {}) {
        if ('string' === typeof source) source = JSON.parse(source);
        this.page_stack = source["page_stack"];
        this.view_tokens = source["view_tokens"];
        this.url_to_page = source["url_to_page"];
    }
}
