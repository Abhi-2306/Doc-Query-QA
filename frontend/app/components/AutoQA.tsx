"use client";

import { useEffect, useState } from "react";
import { getAutoQA } from "../../lib/api";

interface Props {
    triggered: boolean;              // true after successful PDF upload
    onQuestionClick: (q: string) => void;  // sends question to ChatBox
}

export default function AutoQA({ triggered, onQuestionClick }: Props) {
    const [questions, setQuestions] = useState<string[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");

    useEffect(() => {
        if (!triggered) return;

        const fetch = async () => {
            setLoading(true);
            setError("");
            try {
                const data = await getAutoQA();
                setQuestions(data.questions);
            } catch (err: any) {
                setError(err.message || "Could not load suggestions");
            } finally {
                setLoading(false);
            }
        };

        fetch();
    }, [triggered]);

    if (!triggered) return null;

    return (
        <div className="w-full max-w-xl mx-auto mt-6">
            <p className="text-sm font-medium text-gray-600 mb-2">Suggested Questions</p>

            {loading && (
                <p className="text-sm text-gray-400">Generating questions...</p>
            )}

            {error && (
                <p className="text-sm text-red-500">{error}</p>
            )}

            {!loading && questions.length > 0 && (
                <div className="flex flex-wrap gap-2">
                    {questions.map((q, i) => (
                        <button
                            key={i}
                            onClick={() => onQuestionClick(q)}
                            className="px-3 py-1.5 bg-gray-100 hover:bg-blue-100 hover:text-blue-700
                         text-gray-700 text-sm rounded-full border border-gray-200
                         hover:border-blue-300 transition-colors"
                        >
                            {q}
                        </button>
                    ))}
                </div>
            )}
        </div>
    );
}