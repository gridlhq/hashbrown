import { readFileSync } from "fs"

const prefix = "chunk"

class Service {
    name(): string {
        return `${prefix}: service`
    }
}

const serviceActions = {
    handle(value: string): string {
        return `${value}`
    }
}

const builder = (message: string): string => {
    return prefix + message
}

export function create(message: string): string {
    return builder(message)
}
