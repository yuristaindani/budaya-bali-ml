

"use server";

export const postChat = async (question: string) => {
  try {
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
    const response = await fetch(`${apiUrl}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    if (!response.ok) {
      return { 
        success: false, 
        message: "Failed to get response" 
      };
    }

    const data = await response.json();

    return {
      success: data.success,
      message: "Success",
      data: {
        content: data.answer,
        sources: data.sources || [],
        language: data.detected_language || "en"
      },
    };
  } catch (error) {
    return { 
      success: false, 
      message: "Error processing request" 
    };
  }
};